import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy
# modification 5
import torch.optim as optim
KL_Loss = nn.KLDivLoss(reduction='batchmean')
LogSoftmax = nn.LogSoftmax(dim=1)
Softmax = nn.Softmax(dim=1)
# Implementation for pFeMe clients

class UserPFML(User):
    def __init__(self, algorithm,device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate):
        super().__init__(algorithm,device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
# modification 6
#         self.p_optimizer= optim.SGD(self.p_model.parameters(), lr = 0.01, momentum=0.9)

        self.p_optimizer = pFedMeOptimizer(self.p_model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

# modification 7
    def p_update_parameters(self, new_params):
        for param , new_param in zip(self.p_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def train(self, epochs):
        CE_LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update
            self.p_model.train()
            self.model.train()
            X, y = self.get_next_train_batch()

            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                self.p_optimizer.zero_grad()
                output = self.model(X)
                output_m = self.p_model(X)
                # print("mean of output",torch.mean(output))
                # print("mean of output_m",torch.mean(output_m))
                # print("----------------------")

                loss = self.loss(output, y) + KL_Loss(Softmax(output), Softmax(output_m.detach()))
                loss_m = self.loss(output_m, y) + KL_Loss(Softmax(output_m), Softmax(output.detach()))

                loss.backward()
                loss_m.backward()
                if i == self.K - 1:
                    GRAD_LOCAL = []
                    for param in self.model.parameters():
                        GRAD_LOCAL.append(param.grad)
                    GRAD_pLOCAL = []
                    for param in self.p_model.parameters():
                        GRAD_pLOCAL.append(param.grad)

                self.local_model_bar, _ = self.optimizer.step(self.local_model)
                self.persionalized_model_bar, _ = self.p_optimizer.step(self.p_local_model)

                # update local weight after finding aproximate theta

            for new_param, localweight, grad in zip(self.local_model_bar, self.local_model, GRAD_LOCAL):
                # localweight.data = localweight.data -  self.learning_rate * self.lamda*(localweight.data - new_param.data)
                localweight.data = localweight.data - self.learning_rate * grad \
                                   - self.learning_rate * self.lamda * (localweight.data - new_param.data)
            for new_param, localweight, grad in zip(self.persionalized_model_bar, self.p_local_model, GRAD_pLOCAL):
                localweight.data = localweight.data - self.learning_rate * grad \
                                   - self.learning_rate * self.lamda * (localweight.data - new_param.data)


        self.update_parameters(self.local_model)
        self.p_update_parameters(self.p_local_model)

