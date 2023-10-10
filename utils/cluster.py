import numpy as np
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import os
from collections import Counter
import os
import sys
sys.path.append('..')
import torch
import os
from easydict import EasyDict as edict
import json
import time
from loguru import logger

def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)

def random_sample(retrieval_code, retrieval_target=None, ratio=1):
    '''
    Random sample num codes from retrieval hash codes
    Args:
        retrieval_code: torch.Tensor, N*K
        retrieval_target: np.ndarray, N
        num: int
    '''
    total_num = retrieval_code.shape[0]
    num = int(total_num * ratio)
    index = np.random.permutation(total_num)[:num]
    sampled_code = retrieval_code[index, :]
    if retrieval_target is not None:
        sampled_target = retrieval_target[index]
        return sampled_code, sampled_target
    return sampled_code

def uniform_sample(retrieval_code, retrieval_target, num_perclass, num_classes):
    '''
    Uniform sample num codes per class from retrieval hash codes
    Args:
        retrieval_code: torch.Tensor, N*K
        retrieval_target: np.ndarray, N
        num_perclass: int
    '''
    code_length = retrieval_code.shape[1]
    total_num = retrieval_code.shape[0]
    total_num_perclass = int(total_num / num_classes)
    index = np.random.permutation(total_num_perclass)[:num_perclass]
    sample_code = torch.zeros((num_classes * num_perclass, code_length))
    sample_target = torch.zeros(num_classes * num_perclass)
    for i in range(num_classes):
        sample_code[num_perclass*i:num_perclass*(i+1), :] = retrieval_code[retrieval_target==i][index, :]
        sample_target[num_perclass*i:num_perclass*(i+1)] = retrieval_target[retrieval_target==i][index]

    return sample_code, sample_target

def cluster_DBSCAN(retrieval_code, sample=False):
    #! DBSCAN
    code = retrieval_code
    code_length = retrieval_code.shape[-1]
    if sample:    
        code = random_sample(code, ratio=0.1)
    Y = 0.5 * (code_length - code @ code.t())
    label_pred = DBSCAN(eps=1, metric='precomputed', n_jobs=-1).fit_predict(Y) #! cluster retrieval
    n_clusters_ = len(set(label_pred))-1
    print('Number of Clusters: {}'.format(n_clusters_))
    centers = torch.zeros((n_clusters_, code_length))

    for i in range(n_clusters_):
        code_i = code[label_pred==i]
        centers[i] = torch.mean(code_i, dim=0).sign()
    return centers, n_clusters_

def Hierarchical_cluster(code, sample=True, ratio=0.1, epsilon=4, ):
    code_length = code.shape[-1]
    if sample:    
        code = random_sample(code, ratio=ratio)
        
    H = 0.5 * (code_length - code @ code.t())
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=code_length//epsilon).fit(H)
    labels_pred = clustering.labels_
    #! 把小于10个的聚类簇看作噪声点
    cnt = Counter(labels_pred)
    valid_cluster = [x for x in cnt.keys() if cnt[x]>10]
    prototypes = torch.zeros((len(valid_cluster), code_length))

    #! 计算每个类簇的质心
    #*---------------------------------------------------------
    for i in range(len(valid_cluster)):
        code_i = code[labels_pred==valid_cluster[i]]
        prototypes[i] = torch.mean(code_i, dim=0).sign()
    
    return prototypes, len(valid_cluster)

# if __name__ == '__main__':
#     device = torch.device('cuda:2')
#     datasets = ['imagenet', 'svhn']
#     basemethod = 'adsh'
#     code_length = 32
#     logger.add('/data2/suqinghang/research-DFIH/utils/Time/Cluster.log', rotation='500 MB', level='INFO')
#     for dataset in datasets:
#         ori_path = None
#         if ori_path is None:
#             with open('/data2/suqinghang/research-DFIH/config/basemethod_path.JSON', 'r') as f:
#                 paths = edict(json.load(f))
#             ori_path = paths[dataset][basemethod][str(code_length)]
        
#         ratios = [0.01, 0.1, 0.5, 1]
#         epsilons = [2, 4, 6, 8]
#         ori_code = torch.load(os.path.join(ori_path, '{}-inc_retrieval_code{}.t'.format(dataset, code_length)))
        
#         logger.info('---------------------------------------------------------------------------')
#         logger.info('dataset:{} basemethod:{} code_length:{} NUM:{}'.format(dataset, basemethod, code_length, ori_code.shape[0]))
#         for i in range(3):
#             for ratio in ratios:
#                 for epsilon in epsilons:
#                     start = time.time()
#                     p, n = Hierarchical_cluster(ori_code, sample=True, ratio=ratio, epsilon=epsilon)
#                     end = time.time()
#                     logger.info('ratio:{} epsilon:{} time:{} N:{}'.format(ratio, epsilon, end-start, n))
