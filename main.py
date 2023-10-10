import sys
sys.path.append('../')
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from loguru import logger

from utils.Dict2Obj import Dict2Obj
from utils.evaluate import test, test_inc

from data import (cifar10_inc, cifar100_inc, imagenet_inc, svhn_inc)

from method.DFIH import DFIH
DATASET = {'cifar-10': cifar10_inc, 'imagenet': imagenet_inc, 'cifar-100':cifar100_inc, 'svhn': svhn_inc}

METHODS = {'dfih':DFIH}
            
def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Hashing Template')

    parser.add_argument('--config', default='config/CIFAR10.yaml', type=str)
    parser.add_argument('--method', default='dfih', type=str,
                        help='Method used to generate the base k hash code.')
    parser.add_argument('--arch', default='alexnet', type=str,
                        help='path of ori model and database codes.')
    parser.add_argument('--code_length', default=32, type=int,
                        help='Length of hash code.')
    parser.add_argument('--num_class_list', default="5,5", type=str,
                        help='Number of original classes.') 
    parser.add_argument('--start_session', default=1, type=int,
                        help='.')    
    parser.add_argument('--num_works', default=None, type=int,
                        help='.')     

    parser.add_argument('--lr', default=None, type=float)  
    parser.add_argument('--lambda_kd', default=None, type=float)    
    parser.add_argument('--lambda_q', default=None, type=float) 
    parser.add_argument('--lambda_proxy', default=None, type=float) 


    #* params for MDSH_KD
    parser.add_argument('--lwf', default=None, type=float, help='')
    parser.add_argument('--mmd', default=None, type=float, help='')
    parser.add_argument('--lwm', default=None, type=float, help='')
    parser.add_argument('--cvs', default=None, type=float, help='')
    parser.add_argument('--code_consistency', default=1, type=float, help='')

    #* paramas for RFIH   
    parser.add_argument('--AIM', default=1, type=float)  
    parser.add_argument('--APM', default=1, type=float)  
    parser.add_argument('--omega', default=None, type=float)    
    parser.add_argument('--sigma', default=None, type=float)  
    parser.add_argument('--div', default=None, type=float) 

    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.')
    parser.add_argument('--project', default='Test', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--comments', default=None, type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    config = Dict2Obj(config)
    if args.method is not None:
        config.method = args.method
    if args.arch is not None:
        config.arch = args.arch
    if args.code_length is not None:
        config.code_length = args.code_length
    if args.num_class_list is not None:
        config.num_class_list = list(map(int, args.num_class_list.split(',')))
        config.total_session = len(config.num_class_list)

    if args.lwf is not None:
        config.lwf = args.lwf   
    if args.lwm is not None:
        config.lwm = args.lwm 
    if args.mmd is not None:
        config.mmd = args.mmd    
    if args.cvs is not None:
        config.cvs = args.cvs  
    if args.code_consistency is not None:
        config.code_consistency = args.code_consistency 


    if args.lr is not None:
         config.lr = args.lr   
    if args.lambda_kd is not None:
         config.lambda_kd = args.lambda_kd   
    if args.lambda_q is not None:
         config.method_parameters[config.method].lambda_q = args.lambda_q  
    if args.lambda_proxy is not None:
         config.method_parameters[config.method].lambda_proxy = args.lambda_proxy  

    if args.start_session is not None:
         config.start_session = args.start_session  

    if args.AIM is not None:
         config.AIM = args.AIM  
    if args.APM is not None:
         config.APM = args.APM  
    if args.omega is not None:
         config.omega = args.omega   
    if args.sigma is not None:
         config.sigma = args.sigma 
    if args.div is not None:
         config.div = args.div 


    if args.num_works is not None:
        config.num_works = args.num_works
    if args.gpu is not None:
         config.gpu = args.gpu   
    if args.project is not None:
        config.project = args.project
    if args.comments is not None:
        config.comments = args.comments
    if args.save:
        config.save_checkpoint = True

    config.device = torch.device('cpu') if config.gpu is None else torch.device('cuda:{}'.format(config.gpu))

    return config

if __name__ == '__main__':

    torch.set_num_threads(1)
    config = load_config()
    timestr = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) 
    save_name = '{}bits_{}'.format(config.code_length, timestr)
    logger.add(os.path.join('Results', config.project, 'logs', config.method, config.dataset, save_name+'.log'), rotation='500 MB', level='INFO')

    # a = vars(config)
    logger.info('--------------------------Current Settings--------------------------')
    config.method_parameters = config.method_parameters[config.method]
    for key, value in config.items():
        logger.info('{} = {}'.format(key, value))

    dataloader = DATASET[config.dataset]
    for s in range(config.total_session):
        session_id = s
        logger.info('------------Session {} Training Start-------------------------------'.format(session_id))
        
        num_ori_class = sum(config.num_class_list[:session_id])
        num_inc_class = config.num_class_list[session_id]
        # load data
        #* Load dataloader for current session
        train_dataloader, query_dataloader, retrieval_dataloader = dataloader.load_data(
            root               = config.root,
            batch_size         = config.batch_size,
            num_workers        = config.num_works,
            is_original        = False,
            num_origin_classes = num_ori_class,
            num_inc_classes    = num_inc_class,
        )
        inc_labels = query_dataloader.dataset.category_list
        logger.info('Train Size = {}'.format(len(train_dataloader.dataset)))
        logger.info('Query Size = {}'.format(len(query_dataloader.dataset)))
        logger.info('Database Size = {}'.format(len(retrieval_dataloader.dataset)))
        
        #* load method
        if session_id == 0:
            method = METHODS[config.method](config)
        else:
            method.update(session_id=session_id, old_model=prev_model, old_codes=prev_gallery_codes)

        #* Train
        best_mAP = 0.0
        max_iters = method.max_iters
        for i in range(max_iters):
            if session_id == 0:
                model = method.train_iter(train_dataloader, i)
            else:
                model = method.inc_train_iter(train_dataloader, i)

        #* test and save
        if session_id == 0:
            mAP, query_codes, gallery_codes = test(
                model                = model,
                query_dataloader     = query_dataloader,
                retrieval_dataloader = retrieval_dataloader,
                code_length          = config.code_length,
                topk                 = config.topk,
                device               = config.device
            )
            inc_map, ori_map = 0.0, 0.0
            gallery_targets = retrieval_dataloader.dataset.get_onehot_targets()
        else:
            _, ori_query_dataloader, _ = dataloader.load_data(
                root               = config.root,
                batch_size         = config.batch_size,
                num_workers        = 8,
                is_original        = True,
                num_origin_classes = num_ori_class,
                num_inc_classes    = num_inc_class,
            )
            ori_labels = ori_query_dataloader.dataset.category_list
            inc_map, ori_map, mAP, query_code, query_target, gallery_codes, gallery_targets = test_inc(
                model=model,
                ori_query_dataloader=ori_query_dataloader,
                inc_query_dataloader=query_dataloader,
                inc_retrieval_dataloader=retrieval_dataloader,
                ori_retrieval_code=prev_gallery_codes,
                ori_retrieval_target=prev_gallery_targets,
                code_length=config.code_length,
                topk=config.topk,
                device=config.device,
                ismultilabel= config.is_multilabel,
                ori_labels=ori_labels,
                inc_labels=inc_labels
            )

        best_mAP = mAP
        best_model = model
        best_gallery_codes = gallery_codes

        if config.save_checkpoint:
            save_dir = os.path.join('Results', config.project, 'checkpoints', config.method, config.dataset, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  
            torch.save(query_code.cpu(), os.path.join(save_dir, 's{}_query_code{}.t'.format(session_id, config.code_length)))
            torch.save(best_gallery_codes.cpu(), os.path.join(save_dir, 's{}_retrieval_code{}.t'.format(session_id, config.code_length)))
            torch.save(query_dataloader.dataset.get_onehot_targets(), os.path.join(save_dir, 's{}_query_targets{}.t'.format(session_id, config.code_length)))
            torch.save(retrieval_dataloader.dataset.get_onehot_targets(), os.path.join(save_dir, 's{}_retrieval_targets{}.t'.format(session_id, config.code_length)))
            torch.save(model.cpu(), os.path.join(save_dir, '{}_{}.t'.format(config.arch, config.code_length)))
            model.to(config.device)

        logger.info('[it:{}/{}][ORI MAP:{:.4f}][INC MAP:{:.4f}][Overall MAP:{:.4f}][Best MAP:{:.4f}]'.format(
                        i+1, max_iters, ori_map, inc_map, mAP, best_mAP))

        prev_model = best_model
        prev_gallery_codes = best_gallery_codes
        prev_gallery_targets = gallery_targets

    if config.save_checkpoint:
        logger.info('Best model saved at {}'.format(save_dir))
            
