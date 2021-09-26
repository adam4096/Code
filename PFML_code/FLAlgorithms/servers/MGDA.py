import copy
import torch
from torch import nn
from FLAlgorithms.servers.FW_Solver import MinNormSolver



def FedAvg_MOM(w):

    com_grad = copy.deepcopy(w[0])

    for k in com_grad.keys():# 每一层的权重
        W_list = []
        for i in range(len(w)):
            W_list.append(list(w[i][k]))
        alpha, _ = MinNormSolver.find_min_norm_element(W_list)
        temp = 0
        for i in range(len(w)):
            wl = torch.stack(W_list[i],dim=0)
            temp = temp + alpha[i] *wl
        com_grad[k] = temp
    return com_grad