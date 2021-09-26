import torch
import os
import torch.nn.functional as F
from FLAlgorithms.users.userPFML import UserPFML
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
from FLAlgorithms.servers.MGDA import FedAvg_MOM

class PFML(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserPFML(algorithm,device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating PFML server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def aggregate_mgda(self):
        assert (self.users is not None and len(self.users) > 0)
        global_weight = self.model.state_dict()
        temp_grad = copy.deepcopy(self.model)

        total_train = 0
        #if(self.num_users = self.to)
        grad_all = []
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:

            for server_param, user_param ,grad_parm in zip(self.model.parameters(), user.get_parameters(), temp_grad.parameters()):
                grad_parm.data = server_param.data.clone() - user_param.data.clone()
            grad_all.append(temp_grad.state_dict())

        com_grad = FedAvg_MOM(grad_all)
        for k in global_weight.keys():
            global_weight[k] = global_weight[k] - com_grad[k]
        # reload global weight
        self.model.load_state_dict(global_weight)

    def aggregate_loss_based(self,loss_w):
        assert (self.users is not None and len(self.users) > 0)

        for user in self.selected_users:
            # for server_param, user_param ,w in zip(self.model.parameters(), user.get_grads(), loss_w):
                # server_param.data = server_param.data -  user_param.data.clone() * w *20
            for server_param, dif_param ,w in zip(self.model.parameters(), user.get_model_dif(self.model).parameters(), loss_w):
                server_param.data = server_param.data + dif_param.data.clone() * w *20


    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()


            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples



            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.evaluate_personalized_model()
            # self.aggregate_loss_based(Loss_weight)
            # self.aggregate_mgda()
            self.persionalized_aggregate_parameters()


        #print(loss)
        self.save_results()
        self.save_model()
    
  
