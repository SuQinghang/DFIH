# generate orthogonal  K' bits of K bits

import random
from itertools import combinations
from loguru import logger

import numpy as np
import torch
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
from scipy.special import comb  # calculate combination
from utils.optimizeAccel import Lp_box, cal_hamm
import copy

def judge(code_length):
    result = code_length & code_length-1
    if result==0:
        return True
    else:
        return False

def generate_via_hadamard(code_length, num_class):
    ha_d = hadamard(code_length)   # hadamard matrix 
    ha_2d = np.concatenate((ha_d, -ha_d), 0)  # can be used as targets for 2*d hash bit


    if num_class<=code_length:
        hash_targets = torch.from_numpy(ha_d[0:num_class]).float()
        print('hash centers shape: {}'. format(hash_targets.shape))
    elif num_class>code_length:
        hash_targets = torch.from_numpy(ha_2d[0:num_class]).float()
        print('hash centers shape: {}'. format(hash_targets.shape))

    return hash_targets

def generate_via_bernouli(code_length, num_class):
    hash_targets = []
    a = []  # for sampling the 0.5*code_length 
    b = []  # for calculate the combinations of 51 num_class


    for i in range(0, code_length):
        a.append(i)

    for i in range(0, num_class):
        b.append(i)
        
    for j in range(10000):
        hash_targets = torch.zeros([num_class, code_length])
        for i in range(num_class):
            ones = torch.ones(code_length)
            sa = random.sample(a, round(code_length/2))# 第i个类的hash code中, sa位置-1
            ones[sa] = -1
            hash_targets[i]=ones
        com_num = int(comb(num_class, 2))# C(n, 2)
        c = np.zeros(com_num)
        # 计算hash center之间的min和mean
        for i in range(com_num):
            i_1 = list(combinations(b, 2))[i][0]
            i_2 = list(combinations(b, 2))[i][1]
            TF = torch.sum(hash_targets[i_1]!=hash_targets[i_2])
            c[i]=TF

        if np.mean(c)>=int(code_length / 2):  # guarantee the hash center are far away from each other in Hamming space, 20 can be set as 18 for fast convergence
            print(min(c))
            # print("stop! we find suitable hash centers")
            break

    return hash_targets

def generate_centers(K, num_class):
    if K * 2 >= num_class and judge(K):
        Kbit_centers = generate_via_hadamard(code_length=K, num_class=num_class)   
    else:
        Kbit_centers = generate_via_bernouli(code_length=K, num_class=num_class)
    return Kbit_centers

def generate_centers_MDSH(K, num_classes):
    if K * 2 >= num_classes and judge(K):
        naive_centers = generate_via_hadamard(code_length=K, num_class=num_classes)   
    else:
        naive_centers = generate_via_bernouli(code_length=K, num_class=num_classes)
    # min_, mean_ = cal_min_mean(naive_centers)
    # print('Naive: min:{:.2f}, mean:{:.2f}'.format(min_.item(), mean_.item()))

    d = derive_d(K, num_classes)
    if d is None:
        return "Not find minimal distance d."
    print('Minimal separate distance:', d)
    centers = AOP(naive_centers.numpy(), d)
    return centers

def derive_d(K, C):
    '''
    Deriving the minimal distance d
    K: code length
    C: number of class
    '''
    e = 2**K / C
    d = None
    for d_star in range(1, K+1):
        a = sum([comb(K, i) for i in range(0, d_star)])
        b = sum([comb(K, i) for i in range(0, d_star-1)])
        if a>=e and b<e:
            d = d_star
    return d

def AOP(B, d):
    rho = 5e-5
    gamma = (1+5**0.5)/2
    error = 1e-5
    (n_class, bit) = B.shape
    # hash centers initialization
    # np.random.seed(80)

    # metric initialization
    best_st, best_mean, best_min, best_var, best_max = cal_hamm(B)
    best_B = copy.deepcopy(B)
    count = 0
    error_index = {}
    logger.info('CSQ Centers: st: {:.2f}, min: {:.2f}, mean: {:.2f}, var: {:.2f}, max: {:.2f}'.format(
                                best_st, best_min, best_mean, best_var, best_max))
    # print(f"best_st is {best_st}, best_min is {str(best_min)}, best_mean is {best_mean}, best_var is {best_var}, best_max is {str(best_max)}")
    best_st = 0
    # print(f"eval st, eval min, eval mean, eval var, eval max")
    best_B = Lp_box(B, best_B, n_class, d, bit, rho, gamma, error, best_st)
    ev_st, ev_mean, ev_min, ev_var, ev_max = cal_hamm(best_B)
    logger.info('MDSH Centers: st: {:.2f}, min: {:.2f}, mean: {:.2f}, var: {:.2f}, max: {:.2f}'.format(
                                ev_st, ev_min, ev_mean, ev_var, ev_max))
    return best_B
