from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from SparseAttack.common.consts import DEVICE
from SparseAttack.common.utils import max_onehot
from SparseAttack.common.cmd_args import cmd_args
from SparseAttack.trainer import PolicyKL
import torch.optim as optim
import pickle
import torch
import random
import numpy as np
from SparseAttack.mha import GraphAttentionEncoder, MLPEncoder, FixingMLPEncoder 
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # Prediction Net
    if(cmd_args.net == 'mlp'):
        print("Using MLP net.")
        score_net = MLPEncoder().to(DEVICE) # Net-input50, Net2-input500
    else:
        print("Using MHA seq net.")
        score_net = GraphAttentionEncoder().to(DEVICE) # need add positional encoding
    # score_net=nn.DataParallel(score_net)

    # Fixing Net
    # fix_net = FixingMLPEncoder().to(DEVICE) # need add positional encoding

    # train
    if cmd_args.phase == 'train':   
        print("In training phase!")     
        optimizer = optim.Adam(list(score_net.parameters()),lr=1e-4) #cmd_args.learning_rate,cmd_args.weight_decay
                            #    weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=240, gamma=0.8)

        # optimizer=nn.DataParallel(optimizer)
        # scheduler=nn.DataParallel(scheduler)

        # optimizer2 = optim.Adam(list(fix_net.parameters()),lr=1e-4)
        # scheduler2 = StepLR(optimizer2, step_size=2000, gamma=0.5)


        # load_path = "/home/ai/workspaces/longkangli/repo/l2stop/lista_stop/saved_model/binet/pe5_mlp/checkpoint/initial_prediciton_net_19.cp" 
        # load_path = cmd_args.save_dir + 'checkpoint/best_checkpoint_newLabel_newBatch_d500_newLoss_MHA_train.cp'  
        load_path = None 
        if load_path:
            checkpoint = torch.load(load_path)
            score_net.load_state_dict(checkpoint['pred_net'])
            optimizer.load_state_dict(checkpoint['optimizer1'])
            scheduler.load_state_dict(checkpoint['scheduler1'])
            # cmd_args.start_epoch = checkpoint['epoch']
            print(f"Loading Checkpoint: {load_path}...\n")

            # load_path = "/home/ai/workspaces/longkangli/repo/l2stop/lista_stop/saved_model/binet/pe5_mlp/checkpoint/initial_fixing_net_1block_19.cp" 
            load_path = "/home/ai/workspaces/longkangli/repo/l2stop/lista_stop/saved_model/binet/pe5_mlp/checkpoint/initial_fixing_net_19.cp"
            checkpoint = torch.load(load_path)
            # fix_net.load_state_dict(checkpoint['fix_net'])
            # optimizer2.load_state_dict(checkpoint['optimizer2'])
            # scheduler2.load_state_dict(checkpoint['scheduler2'])
            # # cmd_args.start_epoch = checkpoint['epoch']
            # print(f"Loading Checkpoint: {load_path}...\n")
            
        else:
            print("No checkpoint loading...\n")

        trainer = PolicyKL(args=cmd_args,
                           score_net=score_net,
                           optimizer=optimizer, 
                           scheduler=scheduler)
        if cmd_args.phase == 'train':
            trainer.train()
        else:
            trainer.train_binet()

    if cmd_args.phase == 'my_test':
        print("In my test!")
        optimizer = optim.Adam(list(score_net.parameters()),
                               lr=1e-4)
                            #    weight_decay=cmd_args.weight_decay)
        scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
        
        if(cmd_args.net == 'mha'):
            load_path = "../saved_model/final/mha/checkpoint/MHA_1.cp"
        else:
            load_path = "../saved_model/final/mlp/checkpoint/MHA_2.cp"
        checkpoint = torch.load(load_path)
        score_net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cmd_args.start_epoch = checkpoint['epoch']
        print(f"Loading Checkpoint: {load_path} with epoch {cmd_args.start_epoch}...\n")
        trainer = PolicyKL(args=cmd_args,
                        score_net=score_net,
                        optimizer=optimizer, scheduler=scheduler)
        # trainer._my_valid()
        trainer._my_valid_2()


