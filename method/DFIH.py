import sys
sys.path.append('.')
from models.model_loader import load_model
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import copy
import os
from utils.cluster import random_sample
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from method.utils import mask_BCESimilarityToConceptTarget
from pytorch_grad_cam import GradCAM
from utils.evaluate import generate_code
from utils.Centers import generate_centers_MDSH
def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]#depth = LATENT_SIZE
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD_loss(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def KDLoss(logits, labels, temperature=2.0):
    assert not labels.requires_grad, "output from teacher(old task model) should not contain gradients"
    # Compute the log of softmax values
    outputs = torch.log_softmax(logits/temperature,dim=1)
    labels  = torch.softmax(labels/temperature,dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

def NSM(feature, weight):
    norms = torch.norm(feature, p=2, dim=-1, keepdim=True)
    nfeat = torch.div(feature, norms)

    norms_c = torch.norm(weight, p=2, dim=-1, keepdim=True)
    ncenters = torch.div(weight, norms_c)
    logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

    return logits

def Center_Loss(x, centers, labels):
    norms = torch.norm(x, p=2, dim=-1, keepdim=True)
    nfeat = torch.div(x, norms)

    norms_c = torch.norm(centers, p=2, dim=-1, keepdim=True)
    ncenters = torch.div(centers, norms_c)
    logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

    loss = F.cross_entropy(logits, labels)
    return loss

def Proxy_Loss(x, proxy, labels, temp=0.2):
    x = F.normalize(x, p=2, dim=-1)
    proxy = F.normalize(proxy, p=2, dim=-1)

    D = F.linear(x, proxy) / temp
    labels /= torch.sum(labels, dim=1, keepdim=True).expand_as(labels)

    xent_loss = torch.mean(torch.sum(-labels * F.log_softmax(D, -1), -1))
    return xent_loss

class Hash_func(nn.Module):
    def __init__(self, model_name, model, N_cls, N_bits):
        super(Hash_func, self).__init__()
        if model_name == 'alexnet':
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
            self.hash_layer = model.hash_layer
            self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.P = nn.Parameter(torch.FloatTensor(N_cls, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))
        
    def forward(self, x, out_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        feat = self.classifier(x)
        x = self.hash_layer(feat)
        if out_features:
            return x, feat
        else:
            return x

class DFIH(object):
    def __init__(self, config):
        self.config = config
        self.arch = config.arch
        self.code_length = config.code_length
        self.device = config.device
        self.num_class_list = config.num_class_list

        self.model = load_model(self.arch, self.code_length).to(self.device)

        # hash centers in the first session are generated by MDSH
        self.hashcenter_s0 = torch.Tensor(generate_centers_MDSH(self.code_length, num_classes=self.num_class_list[0])).to(self.device)
        
        self.lr = config.lr
        self.multi_lr = 0.05
        params_list = [{'params': self.model.feature_layers.parameters(), 'lr': 0.05 * 1e-4}, # 0.05*(args.lr)
                   {'params': self.model.hash_layer.parameters()}]
        self.optimizer = torch.optim.RMSprop(
            params=params_list,
            lr=1e-4,
            weight_decay=1e-5,
        )
        self.max_iters = config.method_parameters.max_iters
        self.lambda_p = config.method_parameters.lambda_p
        self.lambda_proxy = config.method_parameters.lambda_proxy
        self.lambda_q = config.method_parameters.lambda_q
        self.lambda_kd = config.lambda_kd
        self.session_id = 0
        
    def update(self, session_id, old_model, old_codes=None):
        self.session_id = session_id
        self.old_model = old_model.to(self.device)
        self.old_model.eval()
        
        self.model = Hash_func(self.config.arch, copy.deepcopy(old_model), N_cls=self.num_class_list[self.session_id], N_bits=self.code_length).to(self.device)
        params_list = [{'params': self.model.feature_layers.parameters(), 'lr': self.multi_lr*self.lr}, # 0.05*(args.lr)
                   {'params': self.model.hash_layer.parameters()},
                   {'params': self.model.P}]
        self.optimizer = torch.optim.RMSprop(
            params=params_list,
            lr=self.lr,
            weight_decay=1e-5,
        )
        #* params of RFIH
        self.cluster_threshold = self.code_length // self.config.div
        self.old_prototypes, self.n_clusters_ = self.cluster(old_codes)
        self.old_prototypes = self.old_prototypes.to(self.device)

        if 'resnet' in self.config.arch:
            self.cam = GradCAM(model=self.old_model,
                            target_layers=[self.old_model.feature_layers[-2][-1]],
                            use_cuda=False)
        else:
            self.cam = GradCAM(model=self.old_model,
                            target_layers=[self.old_model.features[-3]],
                            use_cuda=False)



    def train_iter(self, train_dataloader, iter, query_dataloader=None, retrieval_dataloader=None):
        '''
        The first session is trained with current SOTA deep hashing methdod: MDSH[CVPR'23]
        '''
        
        for batch, (data, targets, index) in enumerate(train_dataloader):

            data, targets, index = data.to(self.device), targets.to(self.device), index.to(self.device)

            if not self.config.is_multilabel:
                oh_targets = F.one_hot(targets, self.num_class_list[self.session_id]).float()
            else:
                oh_targets = targets

            f = self.model(data)
            center_loss = Center_Loss(f, self.hashcenter_s0, oh_targets)
            p_loss = self.P_loss(f, oh_targets)
            Q_loss = torch.mean((torch.abs(f) - 1.0) ** 2)

            loss = center_loss + self.lambda_p * p_loss + 0.2 * Q_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.debug(
            '[iter:{}][loss:{:.2f}][center loss:{:.2f}][p loss:{:.2f}][Q loss:{:.2f}]'.format(iter + 1, 
                            loss, center_loss, p_loss, Q_loss))
        return self.model
    
    def inc_train_iter(self, train_dataloader, iter, query_dataloader=None, retrieval_dataloader=None):
        
        if iter == 0:
            #* 旧模型生成的新类别的code
            self.old_code_of_inc = generate_code(self.old_model, train_dataloader, self.code_length, self.device).to(self.device)
            self.old_model.eval()
            self.curr_category_list = train_dataloader.dataset.category_list

        self.adjust_learning_rate(epoch=iter)
        for batch, (data, targets, index) in enumerate(train_dataloader):

            data, targets, index = data.to(self.device), targets.to(self.device), index.to(self.device)
            if not self.config.is_multilabel:
                targets = targets - sum(self.num_class_list[:self.session_id])
                oh_targets = F.one_hot(targets, self.num_class_list[self.session_id]).float()
            else:
                num_old = sum(self.num_class_list[:self.session_id])
                oh_targets = targets[:, num_old:num_old + self.num_class_list[self.session_id]]
            
            am_return, unam_idx = None, None
            if self.config.AIM:
                am_return, unam_idx = self.AIM(data, index)
                if am_return is not None:
                    am_idx, mask = am_return[0], am_return[1]
                    if self.config.APM:
                        data[am_idx] = data[am_idx] * (1-mask)

            #* loss on inc_code
            f = self.model(data)
            hp_loss = Proxy_Loss(f, self.model.P, oh_targets)
            Q_loss = torch.mean((torch.abs(f) - 1.0) ** 2)

            #* loss on proxy
            p_norm = F.normalize(self.model.P, p=2, dim=-1)
            old_p_norm = F.normalize(self.old_prototypes, p=2, dim=-1)
            cosine_loss = torch.nn.CosineSimilarity()
            p_differ_loss = F.relu(p_norm @ old_p_norm.t()).mean()
            p_indepen_loss = (F.relu((p_norm @ p_norm.t())) * torch.triu(torch.ones(p_norm.shape[0], p_norm.shape[0]), diagonal=0).to(self.device)).mean()
            pQ_loss = (torch.diag(p_norm @ F.normalize(self.model.P.sign(), p=2, dim=-1).t())).mean()

            #* kd loss
            kd_loss = 0.0
            logits = NSM(f, self.old_prototypes)
            f_old = self.old_model(data)
            num_old = sum(self.num_class_list[:self.session_id])
            if self.config.code_consistency:
                cosine_loss = torch.nn.CosineSimilarity()
                kd_loss += (1 - cosine_loss(f, f_old)).mean()
            elif self.config.lwf:
                logits_old = NSM(f_old, self.old_prototypes)
                kd_loss += KDLoss(logits, logits_old.detach(), temperature=0.05)
            elif self.config.mmd:
                kd_loss += MMD_loss(f, f_old)
                logits_old = NSM(f_old, self.old_prototypes)
                inc_index = targets >= sum(self.num_class_list[:self.session_id]) # batch中新类别的索引
                if inc_index.sum() != 0:
                    # 用新类别的预测概率进行蒸馏
                    kd_loss += KDLoss(logits[inc_index],
                                logits_old[inc_index].detach(), temperature=0.05)
            elif self.config.lwm:
                bs = f_old.shape[0]
                ori_target = [mask_BCESimilarityToConceptTarget(f_old[i].sign()) for i in range(f_old.shape[0])]
                with GradCAM(model=self.old_model, target_layers=[self.old_model.features[-3]], use_cuda=False) as ori_cam:
                    cam_o = ori_cam(input_tensor=data, targets=ori_target)
                targets = [mask_BCESimilarityToConceptTarget(f[i].sign()) for i in range(f.shape[0])]
                with GradCAM(model=self.model, target_layers=[self.model.features[-3]], use_cuda=False) as inc_cam:
                    cam_n = inc_cam(input_tensor=data, targets=targets)
                ori_am = torch.Tensor(cam_o).to(self.device) 
                inc_am = torch.Tensor(cam_n).to(self.device)   
                cam_o = F.normalize(ori_am.view(bs, -1), p=2, dim=-1)
                cam_n = F.normalize(inc_am.view(bs, -1), p=2, dim=-1)
                lwm_loss = (cam_o - cam_n).norm(p=1, dim=1).mean()
                kd_loss += lwm_loss
            elif self.config.cvs:
                #* Neighbor-session model coherence loss_m
                triplet_criterion = nn.TripletMarginLoss(margin=0.1)
                norm_f = F.normalize(f, p=2, dim=1)
                norm_f_old = F.normalize(f_old, p=2, dim=1)
                dist = torch.cdist(norm_f, norm_f_old)
                hardest_neg = [] #mining hard negative
                sort_pos = torch.argsort(dist, dim=1)
                n = targets.shape[0]
                mask = [tl == targets for tl in targets] # 每个类别和batch内的ground truth similarity vector
                for idx in range(n):
                    for pos in sort_pos[idx]:
                        if mask[idx][pos]==0:
                            neg_idx = pos
                            break
                    hardest_neg.append(norm_f_old[neg_idx])
                hardest_neg = torch.stack(hardest_neg)
                loss_m = triplet_criterion(norm_f, norm_f_old, hardest_neg)
                kd_loss += 10 * loss_m

            loss = hp_loss + self.lambda_q * Q_loss + self.lambda_kd * kd_loss \
                    + self.lambda_proxy * (p_differ_loss + p_indepen_loss + pQ_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.debug(
            '[iter:{}][loss:{:.2f}][hp loss:{:.2f}][Q loss:{:.2f}][kd loss:{:.2f}]'.format(
                                iter + 1, loss.item(), hp_loss.item(), Q_loss.item(), kd_loss.item()))
        logger.debug(
            '[iter:{}][p_differ loss:{:.2f}][p_indepen_loss loss:{:.2f}][pQ_loss loss:{:.2f}]'.format(
                                iter + 1, p_differ_loss.item(), p_indepen_loss.item(), pQ_loss.item()))
        return self.model
      
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.7 ** (epoch // 10))
        #for param_group in optimizer.param_groups:
            #param_group['lr'] = lr
        self.optimizer.param_groups[0]['lr'] = self.multi_lr*lr
        self.optimizer.param_groups[1]['lr'] = lr

        return lr

    def P_loss(self, x, labels):
        S = (labels @ labels.t()>0).float()
        H = (self.code_length - x @ x.t())/2
        H_pos = H[S==1]
        loss_p = torch.log(1 + torch.exp((self.code_length - H_pos)/(2 * self.code_length))).sum()/H_pos.shape[0]
        return loss_p
    
    def cluster(self, ori_code):
        sampled_code = random_sample(ori_code, ratio=0.1)
        H = 0.5 * (self.code_length - sampled_code @ sampled_code.t())

        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=self.cluster_threshold).fit(H.cpu())
        labels_pred = clustering.labels_
        
        #! 把小于10个的聚类簇看作噪声点
        cnt = Counter(labels_pred)
        valid_cluster = [x for x in cnt.keys() if cnt[x]>100]
        # n_clusters_ = clustering.n_clusters_
        prototypes = torch.zeros((len(valid_cluster), self.code_length))
        #! 计算每个类簇的质心
        #*---------------------------------------------------------
        for i in range(len(valid_cluster)):
            code_i = sampled_code[labels_pred==valid_cluster[i]]
            prototypes[i] = torch.mean(code_i, dim=0).sign()
        #*---------------------------------------------------------

        return prototypes, len(valid_cluster)

    def AIM(self, data, idx):
        def hdist(a, b):
            return (a.shape[-1] - a @ b.t())/2
        old_codes = self.old_code_of_inc[idx]
        H = hdist(old_codes.sign(), self.old_prototypes)
        am_idx = (H < self.cluster_threshold).nonzero()
        unam_idx = (((H >= self.code_length//2).sum(1))==self.old_prototypes.shape[0]).nonzero()
        
        am_return = None
        if am_idx.shape[0] != 0:
            am_img = data[am_idx[:, 0]]
            corr_codes = self.old_prototypes[am_idx[:, 1]]
            am_return = [am_idx[:, 0], None]
            if self.config.APM:
                mask = self.APM(am_img, corr_codes)
                am_return = [am_idx[:, 0], mask]

        unam_return = None
        if unam_idx.shape[0] != 0: 
            unam_return = unam_idx[:, 0]

        return am_return, unam_return

    def APM(self, am_img, corr_codes):
        corr_targets = [mask_BCESimilarityToConceptTarget(corr_codes[i]) for i in range(corr_codes.shape[0])]
        batch_cam = self.cam(am_img, corr_targets)
        batch_cam = torch.Tensor(batch_cam).to(self.device)

        mask = torch.sigmoid(self.config.omega * (batch_cam - self.config.sigma))
        mask = mask.unsqueeze(1)
        mask = mask.to(self.device)

        return mask
