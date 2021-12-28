import os
import sys
import pdb
import copy
import time
import random
import threading
import atexit
from datetime import datetime
import tensorflow as tf 
import tensorflow.keras.models as tf_models
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.metrics as tf_metrics
import tensorflow.keras.regularizers as tf_regularizers
import numpy as np
from scipy.stats import truncnorm
from scipy import spatial

from .client import Client

from utils.misc import *
from modules.log_manager import LogManager
from modules.data_manager import DataManager
from modules.model_manager import ModelManager
from modules.train_manager import TrainManager

global_updates = []
class Server:

    def __init__(self, opt):
        self.opt = opt
        self.clients = {}
        self.threads = []
        self.updates = []
        self.task_names = []
        

        self.restored_clients = {}
        self.rid_to_cid = {}
        self.cid_to_vectors = {}
        self.cid_to_weights = {}

        self.curr_round = -1
        self.sparsity_cleints = {
            'sigma': [],
            'psi': []
        }


        self.log_manager = LogManager(self.opt) 
        self.data_manager = DataManager(self.opt, self.log_manager)
        self.model_manager = ModelManager(self.opt, self.log_manager)
        self.train_manager = TrainManager(self.opt, self.log_manager)
        self.log_manager.init_state(None)
        self.data_manager.init_state(None)
        self.model_manager.init_state(None)
        self.train_manager.init_state(None)
        self.load_data()
        self.build_network()

        mu,std,lower,upper = 125,125,0,255
        self.rgauss = self.data_manager.rescale(truncnorm((lower-mu)/std,(upper-mu)/std, loc=mu, scale=std).rvs((1,32,32,3)))

        atexit.register(self.atexit)

    def run(self):
        self.log_manager.print('server process has been started')
        self.create_clients()
        self.train_clients()

    def build_network(self):
        #################################################
        if self.opt.base_network in ['alexnet-like']:
            self.global_model = self.model_manager.build_alexnet_decomposed()
        elif self.opt.base_network in ['resnet9']:
            self.global_model = self.model_manager.build_resnet9_decomposed()

        self.sigma = self.model_manager.get_sigma()
        self.psi = self.model_manager.get_psi()
        self.trainables = [sig for sig in self.sigma]
        for psi in self.psi:
            self.trainables.append(psi)
        self.train_manager.set_details({
            'loss_s': self.loss,
            'model': self.global_model,
            'trainables': self.trainables,
            'num_epochs': self.opt.num_epochs_server,
            'batch_size': self.opt.batch_size_server,
        })
        #################################################
        num_connected = int(round(self.opt.num_clients*self.opt.frac_clients))#连接的客户端个数
        if self.opt.base_network in ['alexnet-like']:
            self.restored_clients = {i:self.model_manager.build_alexnet_plain() for i in range(num_connected)}
        elif self.opt.base_network in ['resnet9']:
            self.restored_clients = {i:self.model_manager.build_resnet9_plain() for i in range(num_connected)}

        for rid, rm in self.restored_clients.items():
            rm.trainable = False
        #################################################

    def load_data(self):
        if self.opt.scenario == 'labels-at-server' or self.opt.scenario =='labels-at-all':
            self.x_train, self.y_train, tname = self.data_manager.get_s_server()
        else:
            self.x_train, self.y_train, tname = None, None, None


        # self.x_valid, self.y_valid =  self.data_manager.get_valid()


        self.x_test, self.y_test =  self.data_manager.get_test()
        self.x_test = self.data_manager.rescale(self.x_test)


        # self.x_valid = self.data_manager.rescale(self.x_valid)

        self.train_manager.set_task({
            'task_name': tname,
            'x_test': self.x_test,
            'y_test': self.y_test,
            'x_train': self.x_train,
            'y_train': self.y_train
        })

    def create_clients(self):
        opt_copied = copy.deepcopy(self.opt)
        gpu_ids = np.arange(len(self.opt.gpu.split(','))).tolist()
        # gpu_ids_real = [int(gid) for gid in self.opt.gpu.split(',')]
        if len(tf.config.experimental.list_physical_devices('GPU'))>0:
            # cid_offset = 0
            self.log_manager.print('creating client processes on gpus ... ')
            for i, gpu_id in enumerate(gpu_ids):
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    self.clients[gpu_id] = Client(gpu_id, opt_copied)
        else:
            self.log_manager.print('creating client processes on cpu ... ')
            num_parallel = 5
            self.clients = {i:Client(i, opt_copied) for i in range(num_parallel)}

    def train_clients(self):
        start_time = time.time()
        self.threads = []
        self.updates = []
        cids = np.arange(self.opt.num_clients).tolist()
        num_connected = int(round(self.opt.num_clients*self.opt.frac_clients))
        for curr_round in range(self.opt.num_rounds*self.opt.num_tasks):
            self.curr_round = curr_round
            #####################################
            if self.opt.scenario == 'labels-at-server':
                self.train_global_model()
            elif self.opt.scenario == 'labels-at-all':
                self.train_global_model()
            #####################################  
            connected_ids = np.random.choice(cids, num_connected, replace=False).tolist() # pick clients
            self.log_manager.print('training clients (round:{}, connected:{})'.format(curr_round, connected_ids))
            sigma = [s.numpy() for s in self.sigma]
            psi = [p.numpy() for p in self.psi]
            while len(connected_ids)>0:
                for gpu_id, gpu_client in self.clients.items():
                    cid = connected_ids.pop(0)
                    with tf.device('/device:GPU:{}'.format(gpu_id)):
                        thrd = threading.Thread(target=self.invoke_client, args=(gpu_client, cid, curr_round, sigma, psi))
                        self.threads.append(thrd)
                        thrd.start()
                    if len(connected_ids) == 0:
                        break
                # wait all threads per gpu
                for thrd in self.threads:
                    thrd.join()   
                self.threads = []
            w = self.aggregate(self.updates)
            if curr_round == 0:
                global_updates.append(self.sigma)
            new_weight = [np.zeros_like(l) for l in sigma]
            #w = self.aggregate(self.updates,global_updates[-1])
            #print(len(sigma),len(psi),len(w),len(new_weight),len(global_updates[-1])
            for i in range(9):
                new_weight[i] = self.opt.lambda_sig*w[i] + self.opt.lambda_psi*w[i+9] + self.opt.lambda_global*global_updates[-1][i]
            #print(len(new_weight))
            global_updates.append(new_weight)
            self.set_weights(global_updates[-1])
            #self.set_weights(w)
            self.train_manager.evaluate_after_aggr()


            self.log_manager.save_current_state({
                 'scores': self.train_manager.get_scores()
             })


            self.updates = []
        self.log_manager.print('all clients done')
        self.log_manager.print('server done. ({}s)'.format(time.time()-start_time))
        sys.exit()

    def invoke_client(self, client, cid, curr_round, sigma, psi):
        update = client.train_one_round(cid, curr_round, sigma, psi)
        #print('updtae:',update)
        self.updates.append(update)

    def train_global_model(self):
        self.log_manager.print('training global_model')
        num_epochs = self.opt.num_epochs_server_pretrain if self.curr_round == 0 else self.opt.num_epochs_server
        self.train_manager.train(self.curr_round, self.curr_round, num_epochs)
        self.log_manager.save_current_state({
            'scores': self.train_manager.get_scores()
        })

    def loss(self, x, y):
        x = self.data_manager.rescale(x)
        y_pred = self.global_model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, y_pred) 
        return y_pred, loss

    def aggregate(self, updates):
        return self.train_manager.fedavg(updates)


    def set_weights(self, new_weights):
        if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':
            for i, nwghts in enumerate(new_weights):
                if i<9:
                    self.sigma[i].assign(new_weights[i])
                    self.psi[i].assign(new_weights[i])#
        elif self.opt.scenario == 'labels-at-server':
            for i, nwghts in enumerate(new_weights):
                self.psi[i].assign(new_weights[i])
    
    def average_client_sparsity(self):
        sigma = [u[2] for u in self.updates]
        psi = [u[3] for u in self.updates]
        self.sparsity_cleints['sigma'].append(np.mean(sigma))
        self.sparsity_cleints['psi'].append(np.mean(psi))

    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.log_manager.print('all client threads have been destroyed.' )
