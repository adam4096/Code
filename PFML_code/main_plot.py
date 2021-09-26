#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)

if(0):
    # mnist mclr
    numusers = 10
    num_glob_iters = 600
    dataset = "Mnist"
    local_ep = [10,10,10,10,10]
    lamda = [15]*5
    learning_rate = [0.01]*5
    beta =  [2.0, 2.0, 2.0,2.0,1.0]
    batch_size = [200,200,200,200,200]
    K = [3]*5
    personal_learning_rate = [0.05]*4
    algorithms = [ "PFML_p","PFML","pFedMe_p","pFedMe","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0):
    # mnist DNN
    numusers = 10
    num_glob_iters = 600
    dataset = "Mnist"
    local_ep = [10,10,10,10,10]
    lamda = [30,30,30,30,15]
    learning_rate = [0.01]*5
    beta =  [2.0, 2.0, 2.0,2.0,1.0]
    batch_size = [200,200,200,200,200]
    K = [3]*5
    personal_learning_rate = [0.05]*4
    algorithms = [ "PFML_p","PFML","pFedMe_p","pFedMe","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                                          learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)



if(0):
    #dnn sys
    numusers = 10
    num_glob_iters = 600
    dataset = "Synthetic"
    local_ep = [10]*5
    lamda = [30,30,30,30,15]

    learning_rate = [0.01,0.01,0.01,0.01,0.01]
    beta =  [2.0,2.0,2.0, 2.0,1.0,]
    batch_size = [200,200,200,200,200]
    K = [3]*5
    personal_learning_rate = [0.01,0.01,0.01,0.01,0.01]
    algorithms = [ "PFML","PFML_p","pFedMe_p","pFedMe","FedAvg"]
    plot_summary_one_figure_synthetic_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                                              learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0):
    #mclr
    numusers = 10
    num_glob_iters = 600
    dataset = "Synthetic"
    local_ep = [10]*5
    lamda = [20,20,20,20,15]

    learning_rate = [0.01,0.01,0.01,0.01,0.01]
    beta =  [2.0,2.0,2.0, 2.0,1.0,]
    batch_size = [200,200,200,200,200]
    K = [3]*5
    personal_learning_rate = [0.01,0.01,0.01,0.01]
    algorithms = [ "PFML_p","PFML","pFedMe_p","pFedMe","FedAvg"]
    plot_summary_one_figure_synthetic_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                                              learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0):
    numusers = 5
    num_glob_iters = 100
    dataset = "Cifar10"
    local_ep = [20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15]
    learning_rate = [0.01,0.01,0.01,0.01,0.01,0.01]
    beta =  [1.0,1.0,1.0, 1.0,1.0, 1.0]
    batch_size = [200,200,200,200,200,200]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.01,0.01,0.01,0.01,0.01,0.01]
    algorithms = [ "PFML","PFML_p","Ditto_p","Ditto","pFedMe_p","pFedMe"]
    plot_summary_one_figure_cifar_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(1):
    numusers = 5
    num_glob_iters = 600
    dataset = "Cifar10"
    local_ep = [10,10,10,10,10]
    lamda = [15,15,15,15,15]
    learning_rate = [0.005]*5
    beta =  [1.0, 1.0,1.0, 1.0,1.0]
    batch_size = [200,200,200,200,200]
    K = [3]*5
    personal_learning_rate = [0.001]*4
    algorithms = [ "PFML_p","PFML","pFedMe_p","pFedMe","FedAvg"]
    plot_summary_one_figure_cifar_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                                          learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
