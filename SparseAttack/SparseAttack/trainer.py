import torch
from tqdm import tqdm
import torch.nn.functional as F
from SparseAttack.common.consts import DEVICE
from SparseAttack.common.utils import max_onehot
import math

import torch.nn as nn  
import os
import numpy as np 

# from Segmentation.cython.src import lpbox # use the Cython from Segmentation 
import time
from statistics import mean 
import math 
from numpy import linalg as LA
import math 

import glob 


def MyMSE(x_hat, x):
    return torch.sum((x_hat - x) ** 2, dim=-1).view(-1)

def sgm(v, s):
    l = len(v)
    # print(l)
    res = 1
    for i in range(l):
        v[i] = max(1, v[i]+s)
        res = res * v[i]
    # v = sum(v)
    # print(v)
    v = pow(res, 1.0/l)
    v = v -s 
    return v

def readFile(file):
    f = open(file)
    line = f.readline()
    xiters = []
    while line:
        xiter = line.split(',')
        xiter = xiter[1:-1]
        xiter = list(map(float, xiter))
        xiters.append(xiter)
        line = f.readline()
    xiters = np.array(xiters)       # (8000, 500)
    xiterss = np.transpose(xiters)  # (500. 8000)
    # print(xiters.shape)
    # print(f"Readfile: {file} ",xiterss.shape)
    return xiterss

import collections 
def getLabel(dataset):
    label = []
    for i in range(len(dataset)):
        if(dataset[i,-1]>=0.5):
            label.append(1.0)
        else:
            label.append(0.0)
    # print("Label length: ",len(label))
    # tmp = collections.Counter(label)
    # print(tmp)
    label = np.array(label)
    label = label[:, np.newaxis]
    # print("Labels: ", label.shape)
    return label 

def readSol(i):
    f = open(f"../cython_solver/sol/bestSol_{i}.txt")
    print(f"Reading solution: ../cython_solver/sol/bestSol_{i}.txt")
    line = f.readline()
    best_obj = line.split()[2]
    print(best_obj)
    line = f.readline()
    obj = []
    sol = []
    while line:
        if 'Time' in line:
            line = line.split()
            time = float(line[1])
            break 
        line = line.split()
        line = line[1:]
        s = float(line[0])
        if s>0.5:
            s = 1
        else:
            s = 0
        sol.append(s)
        o = float(line[1][5:-1])
        obj.append(o)
        line = f.readline()
    print("Obj and Sol length: ",len(obj), len(sol))
    sol = np.array(sol) 
    sol = sol[:, np.newaxis]
    return best_obj, time, obj, sol   

def getSubset(data, idx, ws):
    # idx = 1 # sub set index
    # r = 10  # remainder
    # step_size = 25
    # subdata = data[:,(idx-1)*500+r:idx*500:step_size] # from (idx-1)*500, to idx*500, every 25 pick 1 value. 
    subdata = data[:,(idx-1)*ws:idx*ws]
    # print("subset size: ",subdata.shape)  # (500, 20),  (500,500)
    return subdata 

def deter_fix(subset, sco_sigmoid):
    subset = subset.cpu().detach().numpy() # 500,1
    sco_sigmoid = sco_sigmoid.cpu().detach().numpy() # 500,1 
    print("In deter fix: ", subset.shape)
    fix_idx = []
    fix_val = []
    rest_idx = []
    f1 = 0
    f0 = 0
    for i in range(len(subset)):
        # print(f"{i}: {subset[i]}, {sco_sigmoid[i]}")
        if subset[i] > 0.9:
            f1 = f1 + 1
            idx.append(i)
            val.append(1)
        elif subset[i] < 0.02:
            f0 = f0 + 1
            idx.append(i)
            val.append(0)
        else:
            rest_idx.append(i)
    print(f"Fixed length: {len(fix_idx)}; Rest length: {len(rest_idx)}")
    return fix_idx, fix_val, rest_idx 

def deter_fix(subset, sco_sigmoid, labels, dataset, left_idx):
    subset = subset.cpu().detach().numpy() # 500,1
    sco_sigmoid = sco_sigmoid.cpu().detach().numpy() # 500,1 
    print("In deter fix: ", subset.shape)
    fix_idx = []
    fix_val = []
    rest_idx = []
    f1 = 0
    f0 = 0
    f1e = 0
    f0e = 0
    for i in range(len(subset)):
        # print(f"{i}: {subset[i]}, {sco_sigmoid[i]}")
        org_idx = left_idx[i]
        if sco_sigmoid[i] > 0.88:
            f1 = f1 + 1
            fix_idx.append(i)
            fix_val.append(1)
            if not dataset[org_idx,-1]==labels[org_idx].item():
                print(f"Index: {org_idx}, logits: {subset[i].item()}, prob: {sco_sigmoid[i].item()}, lable: {labels[org_idx].item()}, lpbox: {dataset[org_idx,-1]}")
                f1e = f1e + 1

        elif sco_sigmoid[i] < 0.010:
            f0 = f0 + 1
            fix_idx.append(i)
            fix_val.append(0)
            if not dataset[org_idx,-1]==labels[org_idx].item():
                print(f"Index: {org_idx}, logits: {subset[i].item()}, prob: {sco_sigmoid[i].item()}, lable: {labels[org_idx].item()}, lpbox: {dataset[org_idx,-1]}")
                f0e = f0e + 1
        else:
            rest_idx.append(i)
    print(f"Fixed length: {len(fix_idx)}; fix_1: {f1}, fix_1_error: {f1e}; fix_0: {f0}, fix_0_error: {f0e}; Rest length: {len(rest_idx)}")
    return fix_idx, fix_val, rest_idx, f1e+f0e 

def deter_fix_2(sco_sigmoid):
    data = sco_sigmoid.cpu().detach().numpy() # 500,1 
    fix_val = []
    f1 = 0
    f0 = 0
    f = 0
    C = 0.9
    for i in range(len(data)):
        if data[i] > C:
            f1 = f1 + 1
            fix_val.append(1.0)

        elif data[i] < 1-C:
            f0 = f0 + 1
            fix_val.append(0.0)
        else:
            fix_val.append(-1.0)
    fix_val = np.array(fix_val)
    return fix_val, f1, f0 


def get_lpbox_info():
    path = "../cython/demo/result/xiter_all_e4.csv"
    print(f"load lpbox result path: {path}")
    f = open(path)
    # line = f.readline()
    line = f.readline()
    info = [] # [instance, obj, iters, time]
    while line:   
        line = line.split(',')
        line = list(map(float, line))
        info.append(line)
        line = f.readline()
    return info 

def cal_obj(all_idx, all_val, obj):
    ret_obj = 0.0
    for i in range(len(all_idx)):
        idx = all_idx[i]
        val = all_val[i]
        ret_obj += val * obj[idx]
    return ret_obj 

def compare(pred, sco_sigmoid, labels, dataset):
    for i in range(250):
        print(f"Index: {i}, logits: {pred[i].item()}, prob: {sco_sigmoid[i].item()}, lable: {labels[i].item()}, lpbox: {dataset[i,-1]}")



class PolicyKL:
    def __init__(self, args, score_net, optimizer, scheduler):#, fix_net, optimizer2, scheduler2
        self.args = args
        self.score_net = score_net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = args.num_epochs
        self.var = self.args.var
        self.start_epoch = self.args.start_epoch
        self.ws = self.args.ws 
        # self.fix_net = fix_net
        # self.optimizer2 = optimizer2
        # self.scheduler2 = scheduler2

    # Fucntions: get fixing vectors  
    def _get_fix_vec(self, input):
        if input.shape[0] <= 20000:
            print("#### this is in IF: input shape < 2w")
            pred, pred_sigmoid = self.score_net(input)
            tmp = pred_sigmoid.cpu().detach().numpy()
            sigm = pred_sigmoid.cpu().detach().numpy()

            vec_ret, f1_ret, f0_ret = deter_fix_2(pred_sigmoid)
        else:
            print("#### this is in ELSE: input shape > 2w")
            # print("#### shape is greater than 20,000.")
            batch_size = 10000
            batches = math.ceil(input.shape[0]/batch_size)
            for i in range(batches):
                if i==batches:
                    end = input.shape[0]
                else:
                    end = batch_size * (i+1)
                start =  batch_size * i
                input_batch = input[start:end]    
                pred, pred_sigmoid = self.score_net(input_batch)
                # pred_sigmoid = pred_sigmoid.cpu().detach().numpy()
                vec, f1, f0 = deter_fix_2(pred_sigmoid)
                # print("this is one batch shape: ", pred_sigmoid.shape)

                if i == 0: 
                    # prediction = pred_sigmoid
                    vec_ret = vec 
                    f1_ret = f1
                    f0_ret = f0 
                else:
                    # prediction = np.concatenate((prediction, pred_sigmoid), axis=0)
                    vec_ret = np.concatenate((vec_ret, vec), 0)
                    f1_ret += f1
                    f0_ret += f0 

            print("prediction shape:", vec_ret.shape)
        return vec_ret, f1_ret, f0_ret 


    def _train_mha_100(self, epoch):
        print("Epoch:%d, Learning rate: %f" % (epoch, self.optimizer.param_groups[0]['lr']))
        # n outputs, n-1 nets
        self.score_net.train()
        loss_list = []   
        files_list = glob.glob('../xiter/*')
        for it in tqdm(files_list):
            total_loss = 0.0 
            best_loss = 0.0
            # generate path
            dataset = readFile(it)
            labels = getLabel(dataset)
            sa, sb = dataset.shape
            # print(f"shape of dataset: {dataset.shape}")
            tmp = np.sum(labels)
            # print("tmp: ", tmp)

            labels = torch.from_numpy(labels.astype(np.float32)).to(DEVICE) # (500)


            # left_idx = np.arange(500)
            # rest_idx = left_idx 
            # all_idx = []
            # all_val = []

            self.optimizer.zero_grad()

            # Method 2: contencate all 10 episodes as 1 batch. 
            cont_set = []
            weight = [] 
            for i in range(1,4):
                subset = getSubset(dataset, i, self.ws)
                cont_set.append(subset)
                weight_cur = sa * [1.0/i] # column size = 500
                weight = weight + weight_cur
            weight = np.array(weight)  
            weight = weight[:, np.newaxis]
            cont_set = np.array(cont_set)
            # cont_set = cont_set[:, np.newaxis]
            a,b,c = cont_set.shape # (10,500,ws)
            # print(cont_set.shape)
            cont_set = cont_set.reshape(a*b, c) # (10*500, ws)
            # print(cont_set.shape)
            
            labels = labels.tile((3,1))  #  (10*500, ws)

            # tmp = np.zeros((a*b, 20, 5))
            # for i in range(a*b):
            #     for j in range(20):
            #         tmp[i, j, :] =  cont_set[i, j:(j+5)]
            # # print(tmp.shape)
            # cont_set = tmp 

            cont_set = cont_set.reshape(a*b, 10, 5) # new Reshape: (10*500, ws) -> (10*ws, 20, 5)
            cont_set = torch.from_numpy(cont_set.astype(np.float32)).to(DEVICE)
            weight = torch.from_numpy(weight.astype(np.float32)).to(DEVICE)
            # print(f"Weight shape: {weight.shape}; input data shape: {cont_set.shape}; label shape: {labels.shape}")

            pred, sco_sigmoid = self.score_net(cont_set)
            # print(f"pre shape: {pred.shape}; sco_sigmoid shape: {sco_sigmoid.shape}.")
            loss_fn = nn.BCEWithLogitsLoss(weight=weight) 
            loss = loss_fn(pred, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            # print(f"Problem [{it}]; Total loss: [{total_loss}];")
            loss_list.append(total_loss)       
            
        return mean(loss_list)

    def _train_mlp_100(self, epoch):
        print("Epoch:%d, Learning rate: %f" % (epoch, self.optimizer.param_groups[0]['lr']))
        # n outputs, n-1 nets
        self.score_net.train()
        loss_list = []   
        files_list = glob.glob('../xiter/*')
        # print(files_list)
        for it in tqdm(files_list):
            total_loss = 0.0 
            best_loss = 0.0
            # generate path
            dataset = readFile(it)
            labels = getLabel(dataset)
            sa, sb = dataset.shape
            labels = torch.from_numpy(labels.astype(np.float32)).to(DEVICE) # (500)

            self.optimizer.zero_grad()

            # Method 2: contencate all 5 episodes as 1 batch. 
            cont_set = []
            weight = [] 
            for i in range(1,4):
                subset = getSubset(dataset, i, self.ws)
                cont_set.append(subset)
                weight_cur = sa * [1.0/i] # column size = 500
                weight = weight + weight_cur
            weight = np.array(weight)  
            weight = weight[:, np.newaxis]
            cont_set = np.array(cont_set)
            a,b,c = cont_set.shape # (10,500,ws)
            # print(a,b,c)
            cont_set = cont_set.reshape(a*b, c) # (10*500, ws)
            labels = labels.tile((3,1))  #  (10*500, ws)
            
            cont_set = cont_set.reshape(a*b, 10, 5) # new Reshape: (10*500, ws) -> (10*ws, 20, 5)
            
            # tmp = np.zeros((a*b, 20, 5))
            # for i in range(a*b):
            #     for j in range(20):
            #         tmp[i, j, :] =  cont_set[i, j:(j+5)]
            # # print(tmp.shape)
            # cont_set = tmp 
            
            cont_set = torch.from_numpy(cont_set.astype(np.float32)).to(DEVICE)
            weight = torch.from_numpy(weight.astype(np.float32)).to(DEVICE)
            # print(f"Weight shape: {weight.shape}; input data shape: {cont_set.shape}; label shape: {labels.shape}\n {cont_set[0]}")

            pred, sco_sigmoid = self.score_net(cont_set)
            # print(f"pre shape: {pred.shape}; sco_sigmoid shape: {sco_sigmoid.shape}.")
            loss_fn = nn.BCEWithLogitsLoss(weight=weight) 
            loss = loss_fn(pred, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            # print(f"Problem [{it}]; Total loss: [{total_loss}];")
            loss_list.append(total_loss)       
            
        return mean(loss_list)

    # train init | single net | including: _train_mha_100, _train_mlp_100  
    def train(self):
        """
        training logic
        :return:
        """
        print("This is in training")
        best_val_loss = None
        progress_bar = tqdm(range(self.start_epoch, self.start_epoch+self.epochs))

        f = open(self.args.save_dir + "log/log.txt", 'a')
        for epoch in progress_bar:
            # train
            if self.args.net == 'mlp':
                train_loss = self._train_mlp_100(epoch)
            else:
                train_loss = self._train_mha_100(epoch)
            f.write(f"For Epoch {epoch}, Training Loss Average=[{train_loss}]\n")
            print(f"For Epoch {epoch}, Training Loss Average=[{train_loss}]")
            # if best_val_loss is None or train_loss < best_val_loss:
            #     print("Save NEW best SeqNet in Training!")
            #     best_val_loss = train_loss
            #     checkpoint =  {
            #         "net": self.score_net.state_dict(),
            #         "optimizer": self.optimizer.state_dict(),
            #         "epoch": epoch 
            #     }
            #     torch.save(checkpoint, self.args.save_dir + 'checkpoint/best_checkpoint_ws100_weighted_MHA_train.cp')

            # validation
            # if epoch >= 100:
            #     f.write(f"In validation epoch [{epoch}]\n")
            #     val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_originalSize(f)  # val loss == obj gap 
            #     f.write(f"For Epoch {epoch}, origianl size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
            #     print(f"For Epoch {epoch}, origianl size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_epoch_cython_generalizesize(1)  # val loss == obj gap, 1w 
                # f.write(f"For Epoch {epoch}, 1 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 1 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_epoch_cython_generalizesize(2)  # val loss == obj gap, 5w 
                # f.write(f"For Epoch {epoch}, 2 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 2 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_epoch_cython_generalizesize(3)  # val loss == obj gap, 10w 
                # f.write(f"For Epoch {epoch}, 3 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 3 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_epoch_cython_generalizesize(4)  # val loss == obj gap, 20w  
                # f.write(f"For Epoch {epoch}, 4 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 4 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

            # if best_val_loss is None or val_loss < best_val_loss: 
            if True:
                print("Save NEW best SeqNet in Validation!")
                # best_val_loss = val_loss
                checkpoint =  {
                "net": self.score_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch 
                }
                torch.save(checkpoint, self.args.save_dir + 'checkpoint/checkpoint_' + str(epoch) + '.cp' )
        f.close()

    # for regular size: 1e4. where ground-truth of lpbox is given.
    def _my_valid(self, f=0):
        self.score_net.eval()
        obj_list = []
        time_list = []
        iter_list = []
        obj_gap_list = []
        x_sol_gap_l1 = []
        x_sol_gap_l2 = []

        lpbox_obj_list = []
        lpbox_time_list = []
        lpbox_iter_list = []

        x_sol_gap_list_1_to_0 = []
        x_sol_gap_list_0_to_1 = []
        x_sol_gap_list_pos = [] 
        x_sol_gap_list_neg = []

        lpbox_info = get_lpbox_info() #[instance, obj, iters, time]
        cnt = 0 
        for it in range(100, 110): #self.args.iters_per_eval
            total_loss = 0.0 

            # it = 200

            solver = lpbox.PyLPboxADMMsolver(1e4, it)
            solver.solve_init()

            dataset = readFile(it) # cost time
            labels = getLabel(dataset)  # cost time()

            max_iter = 30
            window_size = self.ws # 10  
            n = 0
            vec = np.zeros([self.args.col], dtype=np.double)  # 1000 
            time_start = time.time() 
            for i in range(int(max_iter/window_size)):
                print(f"\nRound {i}")
                # use cython to solve
                start = window_size * i
                end = window_size * (i+1)
                ret = solver.solve_iter_l2f(start, end, vec, n)
                if ret:
                    break
                xiters = solver.get_x_iters_2d(self.ws) #  (n, window_size) = (10000, 10)
                print(f"xiters shape: {xiters.shape}")
                a,b = xiters.shape
                tmp = np.zeros((a, 5, 5))
                for k in range(a):
                    for j in range(5):
                        tmp[k, j, :] =  xiters[k, j:(j+5)]
                xiters = tmp 
                
                # use deep nets
                input = torch.from_numpy(xiters.astype(np.float32)).to(DEVICE)
                print(f"##### Input shape: {input.shape}")

                vec, f1, f0 = self._get_fix_vec(input)
                # pred, pred_sigmoid = self.score_net(input)
                # vec, f1, f0 = deter_fix_2(pred_sigmoid)

                n = f1 + f0
                if n<=10:
                    n = 0 

                print(f"In loop [{i+1}], fixed [{n}] elements; fix One: {f1}, fix zero {f0}\n")

            time_end = time.time()
            time_cost = time_end-time_start

            # obj = -1.0 * solver.cal_Obj()
            obj = solver.get_obj()
            print(f"Energy {obj}; Time elapse: {time_cost}")

            lpbox_obj = lpbox_info[it][2]
            lpbox_time = lpbox_info[it][4]
            
            # iter = solver.get_iter()
            x_sol = solver.get_x_sol()
            x_sol_gap_1_to_0 = 0
            x_sol_gap_0_to_1 = 0
            x_sol_gap_pos = 0
            for i in range(len(labels)):
                if x_sol[i] != labels[i]:
                    if labels[i]==0 and x_sol[i]==1:
                        x_sol_gap_0_to_1 += 1 
                    if labels[i]==1 and x_sol[i]==0:
                        x_sol_gap_1_to_0 += 1
                else:
                    x_sol_gap_pos += 1 
            x_sol_gap_list_1_to_0.append(x_sol_gap_1_to_0)
            x_sol_gap_list_0_to_1.append(x_sol_gap_0_to_1)
            x_sol_gap_list_pos.append(x_sol_gap_pos)
            x_sol_gap_list_neg.append(x_sol_gap_1_to_0+x_sol_gap_0_to_1)
            print(f"Error sum: [{x_sol_gap_1_to_0+x_sol_gap_0_to_1}], Correct: [{x_sol_gap_pos}]. Error_1to0: [{x_sol_gap_1_to_0}], Error_0to1: [{x_sol_gap_0_to_1}]")

            gap = -(obj-lpbox_obj)/lpbox_obj
            print(f"Fixed Obj: {obj}; Lpbox Obj: {lpbox_obj}; Obj Gap: [{gap}].")
            print(f"Fixed time: {time_cost}, lpbox time: {lpbox_time}.")
            
            # obj_list.append(obj)
            time_list.append(time_cost)
            # # iter_list.append(iter)
            obj_gap_list.append(gap)
            # # x_sol_gap_l1.append(np.linalg.norm(x_sol-labels, 1))
            # # x_sol_gap_l2.append(np.linalg.norm(x_sol-labels, 2))

            # lpbox_obj_list.append(lpbox_obj)
            # # # lpbox_iter_list.append(lpbox_iter)
            # lpbox_time_list.append(lpbox_time)          

            # if iter==10000 or abs(obj-lpbox_obj)/lpbox_obj >= 0.1:
            #     cnt = cnt + 1
        
        # mean_obj = mean(obj_list)
        mean_obj_gap = mean(obj_gap_list)
        mean_x_sol_gap_list_neg = mean(x_sol_gap_list_neg)
        print(f"\nMean Error: [{mean_x_sol_gap_list_neg}]. List: {x_sol_gap_list_neg}")
        print(f"Mean Obj Gap: [{mean_obj_gap}]. List: {obj_gap_list}")
        mean_time = mean(time_list)
        print(f"Mean Time: [{mean_time}]. List: {time_list}")
        # # mean_iter = mean(iter_list)
        # # mean_x_l1 = mean(x_sol_gap_l1)
        # # mean_x_l2 = mean(x_sol_gap_l2)

        # mean_lpbox_obj = mean(lpbox_obj_list)
        # mean_lpbox_time = mean(lpbox_time_list)
        # # mean_lpbox_iter = mean(lpbox_iter_list)
        # print('\n')
        # # print(f"time: {time_list}, \nMean: {mean_time}\n")
        # # print(f"l2f obj: {obj_list}  \nMean: {mean_obj}\n")
        # # print(f"lpbox obj: {lpbox_obj_list}  \nMean: {mean_lpbox_obj}\n")
        # # print(f"obj gap: {obj_gap_list}, \nMean: {mean_obj_gap}")

        # print(f"\n[Lpbox] Avg: Obj={mean_lpbox_obj}; Time={mean_lpbox_time}")
        # print(f"[l2f]: Obj: {mean_obj}; Obj_gap: {mean_obj_gap}; Time={mean_time}")

        return 0

    # for large size: 5e4, 1e5 etc. where ground-truth of lpbox is not given.
    def _my_valid_2(self, f=0):
        self.score_net.eval()
        obj_list = []
        time_list = []
        iter_list = []
        obj_gap_list = []
        x_sol_gap_l1 = []
        x_sol_gap_l2 = []

        lpbox_obj_list = []
        lpbox_time_list = []
        lpbox_iter_list = []

        x_sol_gap_list_1_to_0 = []
        x_sol_gap_list_0_to_1 = []
        x_sol_gap_list_pos = [] 
        x_sol_gap_list_neg = []

        # lpbox_info = get_lpbox_info() #[instance, obj, iters, time]
        cnt = 0 
        for it in range(10): #self.args.iters_per_eval
            total_loss = 0.0 

            # it = 200

            solver = lpbox.PyLPboxADMMsolver(1e4, it)
            solver.solve_init()

            # dataset = readFile(it) # cost time
            # labels = getLabel(dataset)  # cost time()

            max_iter = 30
            window_size = self.ws # 10  
            n = 0
            vec = np.zeros([self.args.col], dtype=np.double)  # 1000 
            time_start = time.time() 
            for i in range(int(max_iter/window_size)):
                print(f"\nRound {i}")
                # use cython to solve
                start = window_size * i
                end = window_size * (i+1)
                ret = solver.solve_iter_l2f(start, end, vec, n)
                if ret:
                    break
                xiters = solver.get_x_iters_2d(self.ws) #  (n, window_size) = (10000, 10)
                print(f"xiters shape: {xiters.shape}")
                a,b = xiters.shape
                tmp = np.zeros((a, 5, 5))
                for k in range(a):
                    for j in range(5):
                        tmp[k, j, :] =  xiters[k, j:(j+5)]
                xiters = tmp 
                
                # use deep nets
                input = torch.from_numpy(xiters.astype(np.float32)).to(DEVICE)
                print(f"##### Input shape: {input.shape}")

                vec, f1, f0 = self._get_fix_vec(input)
                # pred, pred_sigmoid = self.score_net(input)
                # vec, f1, f0 = deter_fix_2(pred_sigmoid)

                n = f1 + f0
                if n<=10:
                    n = 0 

                print(f"In loop [{i+1}], fixed [{n}] elements; fix One: {f1}, fix zero {f0}\n")

            time_end = time.time()
            time_cost = time_end-time_start

            # obj = -1.0 * solver.cal_Obj()
            obj = solver.get_obj()
            print(f"Energy {obj}; Time elapse: {time_cost}")

            # lpbox_obj = lpbox_info[it][2]
            # lpbox_time = lpbox_info[it][4]
            
            # # iter = solver.get_iter()
            # x_sol = solver.get_x_sol()
            # x_sol_gap_1_to_0 = 0
            # x_sol_gap_0_to_1 = 0
            # x_sol_gap_pos = 0
            # for i in range(len(labels)):
            #     if x_sol[i] != labels[i]:
            #         if labels[i]==0 and x_sol[i]==1:
            #             x_sol_gap_0_to_1 += 1 
            #         if labels[i]==1 and x_sol[i]==0:
            #             x_sol_gap_1_to_0 += 1
            #     else:
            #         x_sol_gap_pos += 1 
            # x_sol_gap_list_1_to_0.append(x_sol_gap_1_to_0)
            # x_sol_gap_list_0_to_1.append(x_sol_gap_0_to_1)
            # x_sol_gap_list_pos.append(x_sol_gap_pos)
            # x_sol_gap_list_neg.append(x_sol_gap_1_to_0+x_sol_gap_0_to_1)
            # print(f"Error sum: [{x_sol_gap_1_to_0+x_sol_gap_0_to_1}], Correct: [{x_sol_gap_pos}]. Error_1to0: [{x_sol_gap_1_to_0}], Error_0to1: [{x_sol_gap_0_to_1}]")

            # gap = -(obj-lpbox_obj)/lpbox_obj
            # print(f"Fixed Obj: {obj}; Lpbox Obj: {lpbox_obj}; Obj Gap: [{gap}].")
            # print(f"Fixed time: {time_cost}, lpbox time: {lpbox_time}.")
            
            obj_list.append(obj)
            time_list.append(time_cost)
            # # iter_list.append(iter)
            # obj_gap_list.append(gap)
            # # x_sol_gap_l1.append(np.linalg.norm(x_sol-labels, 1))
            # # x_sol_gap_l2.append(np.linalg.norm(x_sol-labels, 2))

            # lpbox_obj_list.append(lpbox_obj)
            # # # lpbox_iter_list.append(lpbox_iter)
            # lpbox_time_list.append(lpbox_time)          

            # if iter==10000 or abs(obj-lpbox_obj)/lpbox_obj >= 0.1:
            #     cnt = cnt + 1
        
        # mean_obj = mean(obj_list)
        mean_obj_list = mean(obj_list)
        print(f"Mean Obj: [{mean_obj_list}]. List: {obj_list}")
        mean_time = mean(time_list)
        print(f"Mean Time: [{mean_time}]. List: {time_list}")
        # # mean_iter = mean(iter_list)
        # # mean_x_l1 = mean(x_sol_gap_l1)
        # # mean_x_l2 = mean(x_sol_gap_l2)

        # mean_lpbox_obj = mean(lpbox_obj_list)
        # mean_lpbox_time = mean(lpbox_time_list)
        # # mean_lpbox_iter = mean(lpbox_iter_list)
        # print('\n')
        # # print(f"time: {time_list}, \nMean: {mean_time}\n")
        # # print(f"l2f obj: {obj_list}  \nMean: {mean_obj}\n")
        # # print(f"lpbox obj: {lpbox_obj_list}  \nMean: {mean_lpbox_obj}\n")
        # # print(f"obj gap: {obj_gap_list}, \nMean: {mean_obj_gap}")

        # print(f"\n[Lpbox] Avg: Obj={mean_lpbox_obj}; Time={mean_lpbox_time}")
        # print(f"[l2f]: Obj: {mean_obj}; Obj_gap: {mean_obj_gap}; Time={mean_time}")

        return 0



    def valid(self):
        """
        training logic
        :return:
        """
        best_val_loss = None
        progress_bar = tqdm(range(self.start_epoch, self.start_epoch+1))

        f = open("../saved_model/final/mha/log/log_test.txt", 'a')

        for epoch in progress_bar:
            # # train
            # if self.args.net == 'mlp':
            #     train_loss = self._train_epoch_mlp_100_(epoch)
            # else:
            #     train_loss = self._train_epoch_mha_100_(epoch)
            # f.write(f"For Epoch {epoch}, Training Loss Average=[{train_loss}]\n")
            # print(f"For Epoch {epoch}, Training Loss Average=[{train_loss}]")

            # validation
            if epoch >= 0:
                # f.write(f"In validation epoch [{epoch}]\n")

                self._valid_smallSize(f)

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_originalSize(f)  # val loss == obj gap 
                # # f.write(f"For Epoch {epoch}, origianl size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, origianl size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_generalizeSize(0)  # val loss == obj gap, 1w 
                # # f.write(f"For Epoch {epoch}, 1 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_generalizeSize(2)  # val loss == obj gap, 5w 
                # # f.write(f"For Epoch {epoch}, 2 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 2 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_generalizeSize(3)  # val loss == obj gap, 10w 
                # # f.write(f"For Epoch {epoch}, 3 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 3 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")

                # val_loss, mean_obj_gap, mean_x_sol_gap_1_to_0, mean_x_sol_gap_0_to_1, mean_x_sol_gap_list_pos = self._valid_generalizeSize(4)  # val loss == obj gap, 20w  
                # # f.write(f"For Epoch {epoch}, 4 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
                # print(f"For Epoch {epoch}, 4 generalize size: Vali Loss Average=[{val_loss}], Vali Obj Gap={mean_obj_gap}. Gap_1_to_0={mean_x_sol_gap_1_to_0}, Gap_0_to_1={mean_x_sol_gap_0_to_1}, Same={mean_x_sol_gap_list_pos}\n")
        f.close()
