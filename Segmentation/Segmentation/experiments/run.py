from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Segmentation.common.consts import DEVICE
from Segmentation.common.utils import max_onehot
from Segmentation.common.cmd_args import cmd_args
from Segmentation.trainer import PolicyKL
import torch.optim as optim
import pickle
import torch
import random
import numpy as np
from Segmentation.mha import GraphAttentionEncoder, MLPEncoder, FixingMLPEncoder 
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


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
    fix_net = FixingMLPEncoder().to(DEVICE) # need add positional encoding

    # train
    if cmd_args.phase == 'train':   
        print("In training phase!")     
        optimizer = optim.Adam(list(score_net.parameters()),lr=1e-4)
        scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
  
        load_path = None 
        if load_path:
            checkpoint = torch.load(load_path)
            score_net.load_state_dict(checkpoint['pred_net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # cmd_args.start_epoch = checkpoint['epoch']
            print(f"Loading Checkpoint: {load_path}...\n")
            checkpoint = torch.load(load_path)
            fix_net.load_state_dict(checkpoint['fix_net'])
            
        else:
            print("No checkpoint loading...\n")

        trainer = PolicyKL(args=cmd_args,
                           score_net=score_net,
                           optimizer=optimizer, 
                           scheduler=scheduler)
        trainer.train()

    if cmd_args.phase == 'test':
        print("In my test!")
        optimizer = optim.Adam(list(score_net.parameters()),
                               lr=1e-4)
                            #    weight_decay=cmd_args.weight_decay)
        scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
        
        if(cmd_args.net == 'mha'):
            load_path = "../saved_model/mha/checkpoint/checkpoint_5.cp"
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
        trainer._my_valid()
        # trainer._my_valid_2()


