#!/usr/bin/python
# -*- coding: UTF-8 -*-
from skimage.segmentation import slic
from PIL import Image
import numpy as np
import time
import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils import *
from flags import parse_handle
from model import CifarNet
import glob  

# set random seed for reproduce
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

#parsing input parameters
parser = parse_handle()
args = parser.parse_args()

#settings
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
DEVICE = torch.device('cuda')

# mean and std, used for normalization
img_mean = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).astype('float32')
img_std = np.array([1, 1, 1]).reshape((1, 3, 1, 1)).astype('float32')
img_mean_cuda = torch.from_numpy(img_mean).to(DEVICE)
img_std_cuda = torch.from_numpy(img_std).to(DEVICE)
img_normalized_ops = (img_mean_cuda, img_std_cuda)

# --attacked_model cifar_best.pth --img_file img0.png --target 1 --k 200

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
    # print("Readfile:",xiterss.shape)
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

def main():
    args.attacked_model = "cifar_best.pth"
    args.k = 200 

    files_list = glob.glob('data/*')
    # print(files_list)

    gg = open('result/all_mha.csv', 'a+')

    files_list = files_list[36:]
    print(files_list)
    for i in files_list:
        t = i.split('/')[-1]
        t = t.split('_')[0]
        args.target = int(t)-2  
        if args.target < 0:
            args.target += 9
        print(args.target)

        single_image(i, gg)
    
    gg.close() 


def single_image(file, gg):
    start = time.time()

    # define model and move it cuda
    model = CifarNet().eval().cuda()
    model.load_state_dict(torch.load(args.attacked_model))

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # training
    batch_train(model, file, gg)

    end = time.time()
    print(f"cost time for process: {end-start}")

def batch_train(model, img_file, gg):  
    num_success = 0.0
    counter =0.0
    L0 = 0.0
    L1 = 0.0
    L2 = 0.0
    Li = 0.0
    WL1 = 0.0
    WL2 = 0.0
    WLi = 0.0

    cur_start_time = time.time()
    # load image and preprocessing
    print('Image:{}'.format(img_file))
    input_image = Image.open(img_file)
    input_image = input_image.resize((args.img_resized_width, args.img_resized_height))
    
    #calculate mask for group sparsity
    image_4_mask = np.array(input_image, dtype=np.uint8)  
    segments = slic(image_4_mask, n_segments=args.segments, compactness=10)  
    
    #axis transpose, rescaled to [0,1] and normalized
    input_image = np.array(input_image, dtype=np.float32)  
    if input_image.ndim <3:                               
        input_image = input_image[:,:,np.newaxis]
        
    
    input_image = np.transpose(input_image, (2, 0, 1))     
    input_image = input_image[np.newaxis,...]              
    input_image = input_image / 255              
    scaled_image = torch.from_numpy(input_image).cuda() 
    
    #process block mask
    block_num = max(segments.flatten())+1 - min(segments.flatten())     
    B = np.zeros((block_num, input_image.shape[1], args.img_resized_width, args.img_resized_height)) 

    label_gt = int(torch.argmax(model(scaled_image-0.5)).data)
    label_target = args.target
    # assert label_gt != label_target, 'Target label and ground truth label are same, choose another target label.'
    print('Origin Label: {}, Target Label: {}'.format(label_gt, label_target))

    for index in range(min(segments.flatten()),max(segments.flatten())+1):
        mask = (segments == index)
        B[index,:,mask] = 1
    B = torch.from_numpy(B).cuda().float()        
    noise_Weight = compute_sensitive(scaled_image, args.weight_type)      
    # print('target sparse k : {}'.format(args.k))
    
    #train
    img_file = img_file.split('/')[-1]
    results = train_adptive(int(0), model, scaled_image, label_target, B, noise_Weight, img_file)
    results['args'] = vars(args)  
    results['img_name'] = img_file
    results['running_time'] = time.time() - cur_start_time
    results['ground_truth'] = label_gt
    results['label_target'] = label_target
    results['segments'] = segments.tolist()
    results['noise_weight'] = noise_Weight.cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist()  
 
    # logging brief summary
    counter +=1
    if results['status'] == True:
        num_success = num_success + 1
    
    # statistic for norm
    L0 += results['L0']
    L1 += results['L1']
    L2 += results['L2']
    Li += results['Li']
    WL1 += results['WL1']
    WL2 += results['WL2']
    WLi += results['WLi']
    
    # save metaInformation and results to logfile
    # save_results(results, args)
    
    print('#'*30)
    print('image=%s, clean-img-prediction=%d, target-attack-class=%d, adversarial-image-prediction=%d' \
            %(results['img_name'], label_gt,label_target,results['noise_label'][0]))
    print('statistic information: success-attack-image/total-attack-image= %d/%d, attack-success-rate=%f, L0=%f, L1=%f, L2=%f, L-inf=%f' \
            %(num_success, counter , num_success/counter, L0/counter, L1/counter, L2/counter, Li/counter))
    print('#'*30+'\n'*2)

    gg.write(f"Img:{results['img_name']}, Origin Label:{label_gt}, Target Label:{label_target}, asr:{num_success}, L0:{L0}, L1:{L1}, L2:{L2}, L-inf:{Li}\n")

    # g = open('log.txt', 'a+')
    # g.write('image=%s, clean-img-prediction=%d, target-attack-class=%d, adversarial-image-prediction=%d\n' \
    #         %(results['img_name'], label_gt,label_target,results['noise_label'][0]))
    # g.write('statistic information: success-attack-image/total-attack-image= %d/%d, attack-success-rate=%f, L0=%f, L1=%f, L2=%f, L-inf=%f\n' \
    #         %(num_success, counter , num_success/counter, L0/counter, L1/counter, L2/counter, Li/counter))
    # g.close() 


def train_adptive(i, model, images, target, B, noise_Weight, img_file):
    args.lambda1 = args.init_lambda1
    lambda1_upper_bound = args.lambda1_upper_bound
    lambda1_lower_bound = args.lambda1_lower_bound
    results_success_list=[]
    args.lambda1_search_times = 6
    print(f"there are will be {args.lambda1_search_times} loops for search.\n")
    img_file = img_file.split('.')[0]
    # print("Test before search loop. ", img_file)
    for search_time in range(1, args.lambda1_search_times+1):
        f = None 
        # f = open(f"xiters_init0/{img_file}_{search_time}.csv", 'w+')
        results = train_sgd_atom(model, images, target, B, noise_Weight, f)
        # f.close()
        results['lambda1'] = args.lambda1

        if results['status'] == True:
            results_success_list.append(results)
            
        if search_time < args.lambda1_search_times:
            if results['status'] == True:
                if args.lambda1 < 0.01*args.init_lambda1:  
                    break
                # success, divide lambda1 by two
                lambda1_upper_bound = min(lambda1_upper_bound,args.lambda1)
                if lambda1_upper_bound < args.lambda1_upper_bound:
                    args.lambda1 = (lambda1_upper_bound+ lambda1_lower_bound)/2
            else:
                # failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lambda1_lower_bound = max(lambda1_lower_bound, args.lambda1)
                if lambda1_upper_bound < args.lambda1_upper_bound:
                    args.lambda1 = (lambda1_upper_bound+ lambda1_lower_bound)/2
                else:
                    args.lambda1 *= 10
        # print(results)
    
    # if succeed, return the last successful results  
    if results_success_list:       
        return results_success_list[-1]
    # if fail, return the current results 
    else:
        return results

        
def train_sgd_atom(model, images, target_label, B, noise_Weight, f):
    target_label_tensor=torch.tensor([target_label]).cuda()

    # G = torch.zeros(images.shape, dtype=torch.float32).cuda()
    G = torch.ones(images.shape, dtype=torch.float32).cuda()
    epsilon = torch.zeros(images.shape, dtype=torch.float32).cuda()
    
    cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
    ori_prediction, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)  
    
    cur_lr_e = args.lr_e
    cur_lr_g = {'cur_step_g': args.lr_g, 'cur_rho1': args.rho1, 'cur_rho2': args.rho2, 'cur_rho3': args.rho3,'cur_rho4': args.rho4}
    for mm in range(1,args.maxIter_mm+1): 
        start = time.time()  
        epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, mm, False)
        end1 = time.time()
        print(f"cost time for epsilon update: {end1-start}\n")

        # G, cur_lr_g = update_G(model, images, target_label_tensor, epsilon, G, cur_lr_g, B, noise_Weight, mm, f)
        G, cur_lr_g = update_G_l2f(model, images, target_label_tensor, epsilon, G, cur_lr_g, B, noise_Weight, mm, f=None)
        end2 = time.time()
        print(f"cost time for G update: {end2-end1}\n")
    
    G = (G > 0.5).float()
    epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, mm, True)  
    
    
    cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
    noise_label, adv_image = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
    
    # recording results per iteration
    if noise_label[0] == target_label:
        results_status=True
    else:
        results_status=False  

    results = {
        'status': results_status,
        'noise_label': noise_label.tolist(),
        'ori_prediction': ori_prediction.tolist(),
        'loss': cur_meta['loss']['loss'],
        'l2_loss': cur_meta['loss']['l2_loss'],
        'cnn_loss': cur_meta['loss']['cnn_loss'],
        'group_loss':cur_meta['loss']['group_loss'],
        'G_sum': cur_meta['statistics']['G_sum'],
        'L0': cur_meta['statistics']['L0'],
        'L1': cur_meta['statistics']['L1'],
        'L2': cur_meta['statistics']['L2'],
        'Li': cur_meta['statistics']['Li'],
        'WL1': cur_meta['statistics']['WL1'],
        'WL2': cur_meta['statistics']['WL2'],
        'WLi': cur_meta['statistics']['WLi'],
        'G' : G.detach().cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist(),
        'epsilon' : epsilon.detach().cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist(),
        'adv_image' : adv_image.detach().cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist()
    }
    return results

def update_epsilon(model, images, target_label, epsilon, G, init_lr, B, noise_Weight, out_iter, finetune):
    cur_step = init_lr
    train_epochs = int(args.maxIter_e/2.0) if finetune else args.maxIter_e
 
    for cur_iter in range(1,train_epochs+1): 
        epsilon.requires_grad = True  
        G.requires_grad = False
        
        images_s = images + torch.mul(epsilon, G) 
        images_s = torch.clamp(images_s, args.min_pix_value, args.max_pix_value)  
        images_s = Normalization(images_s, img_normalized_ops) 
        prediction = model(images_s)
        
        #loss
        if args.loss == 'ce':
            ce = nn.CrossEntropyLoss()
            loss = ce(prediction, target_label)  
        elif args.loss == 'cw':     
            label_to_one_hot = torch.tensor([[target_label.item()]])
            label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
            
            real = torch.sum(prediction*label_one_hot)
            other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction - (label_one_hot*10000))
            loss = torch.clamp(other_max - real + args.confidence, min=0)
            

        if epsilon.grad is not None:
            epsilon.grad.data.zero_()
        loss.backward(retain_graph=True)
        epsilon_cnn_grad = epsilon.grad

        epsilon_grad = 2*epsilon*G*G*noise_Weight*noise_Weight + args.lambda1 * epsilon_cnn_grad
        epsilon = epsilon - cur_step * epsilon_grad
        epsilon = epsilon.detach()  
        
        # updating learning rate
        if cur_iter % args.lr_decay_step == 0:
            cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
                   
        # tick print
        if cur_iter % args.tick_loss_e == 0:
            cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
            noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)  
        
    return epsilon, cur_step


from SparseAttack.mha import GraphAttentionEncoder, MLPEncoder
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# import torch.nn as nn
# def load_l2f(net, load_path):
#     if(net == 'mlp'):
#         print("Using MLP net.")
#         score_net = MLPEncoder().to(DEVICE) # Net-input50, Net2-input500
#     else:
#         print("Using MHA seq net.")
#         score_net = GraphAttentionEncoder().to(DEVICE) # need add positional encoding
#     # optimizer = optim.Adam(list(score_net.parameters()),lr=1e-4) 
#     # scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
#     checkpoint = torch.load(load_path)
#     # score_net.load_state_dict(checkpoint['net'])
#     # optimizer.load_state_dict(checkpoint['optimizer'])

#     return score_net 

# this is the lpbox ADMM goes. 
def update_G_l2f(model, images, target_label, epsilon, G, init_params, B, noise_Weight, out_iter, f):
    # initialize learning rate
    cur_step = init_params['cur_step_g']
    cur_rho1 = init_params['cur_rho1']
    cur_rho2 = init_params['cur_rho2']
    cur_rho3 = init_params['cur_rho3']
    cur_rho4 = init_params['cur_rho4']

    # print(f"G shape: {G.shape}")
    # print(f"Image shape: {images.shape}")

    # initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros matrix
    y1 = torch.ones_like(G)
    y2 = torch.ones_like(G)
    y3 = torch.ones_like(G)
    z1 = torch.zeros_like(G)
    z2 = torch.zeros_like(G)
    z3 = torch.zeros_like(G)
    z4 = torch.zeros(1).cuda()
    ones = torch.ones_like(G)

    other_params = {}
    other_params['y1'] = y1
    other_params['y2'] = y2
    other_params['y3'] = y3
    other_params['z1'] = z1
    other_params['z2'] = z2
    other_params['z3'] = z3
    other_params['z4'] = z4
    other_params['ones'] = ones 

    net = 'mha'
    if net=='mlp':
        load_path = "saved_model/mlp/checkpoint/checkpoint_19.cp"
    else:
        load_path = "saved_model/mha/checkpoint/checkpoint_19.cp"

    print(f"Load path...{load_path}")
    G_permu = None 
    C = 0.90

    if(net == 'mlp'):
        # print("Using MLP net.")
        score_net = MLPEncoder().to(DEVICE) # Net-input50, Net2-input500
    else:
        # print("Using MHA seq net.")
        score_net = GraphAttentionEncoder().to(DEVICE) # need add positional encoding
    checkpoint = torch.load(load_path)
    score_net.load_state_dict(checkpoint['net'])

    total = 3*32*32
    for i in range(3):
        start_iter = i * 50
        end_iter = (i+1) * 50
        if G_permu is None:
            init_params, other_params, G_permu = loop(model, images, target_label, epsilon, G, init_params, other_params, B, noise_Weight, start_iter, end_iter)
        else:
            # print("test 90: ", torch.cuda.is_available())
            
            # input = G_permu  # G_permu (3,32,32,25) -> (3,32,32,20,5)。 G (1,3,32,32) 
            n,c,w,h = G_permu.shape  
            G_permu = G_permu.reshape(n*c*w, h) # (3,32,32,25) -> (3*32*32, 25) 
            G_tmp = G_permu.cpu().detach().numpy()
            # tmp = np.zeros((n*c*w, 10, 5))
            # tmp = torch.zeros((n*c*w, 20, 5)).cuda()
            # for i in range(n*c*w):
            #     for j in range(20):
            #         tmp[i, j, :] =  G_tmp[i, j:(j+5)]
            tmp = G_tmp.reshape(n*c*w, 10, 5) # new Reshape: (10*500, ws) -> (10*ws, 20, 5)
            # print(tmp) 

            input = torch.from_numpy(tmp.astype(np.float32)).to(DEVICE)
            # print("Test 99: ",input.shape)
            pred, pred_sigmoid = score_net(input) # (3*32*32, 20, 5) -> (3*32*32, 1)
            # print(pred_sigmoid.flatten())

            # test = pred_sigmoid.cpu().detach().numpy().flatten() 
            # print(test.shape) 
            # tmp = collections.Counter(test)
            # print("test 101",tmp)

            # a = list(tmp.keys()) 
            # a.sort()
            # print(a) 


            # dataset = readFile(f'xiters_init1/7_83_1.csv')
            # label = getLabel(dataset)
            # for i in range(n*c*w):
            #     if label[i]==1:
            #         print(i)
            #         t = i
            #         break
            # print(input[t], pred_sigmoid[t])

            # break  

            # print("Test 100: ",pred_sigmoid.shape) 
            cnt1 = 0
            cnt0 = 0
            for i in range(n*c*w):
                if pred_sigmoid[i] > C:
                    # pred_sigmoid[i] = 1.0
                    pred_sigmoid[i] = 1.0
                    cnt1 += 0
                elif pred_sigmoid[i] < 1-C:
                    pred_sigmoid[i] = 0.0
                    cnt0 += 1 
                else:
                    pred_sigmoid[i] = G_permu[i,-1]
            # print(f'Test 101: {pred_sigmoid.shape}')
            G = pred_sigmoid.reshape(n, c, w, 1)  # (3*32*32, 1) -> (3,32,32,1
            # print(f'Test 102:  the shape of G is {G.shape}')
            G = G.permute(3,0,1,2).detach() 
            # print(f'Test 103:  the shape of G is {G.shape}')
            total = total - (cnt1 + cnt0) 
            print(f"Fixed [{cnt1}]+[{cnt0}]=[{cnt1 + cnt0}] elements. Left [{total}] elements")

            
            init_params, other_params, G_permu = loop(model, images, target_label, epsilon, G, init_params, other_params, B, noise_Weight, start_iter, end_iter)

    # res_param = {'cur_step_g': cur_step, 'cur_rho1': cur_rho1,'cur_rho2': cur_rho2, 'cur_rho3': cur_rho3,'cur_rho4': cur_rho4}
    res_param = init_params 
    return G, res_param


def loop(model, images, target_label, epsilon, G, init_params, other_params, B, noise_Weight, start_iter, end_iter):

    cur_step = init_params['cur_step_g']
    cur_rho1 = init_params['cur_rho1']
    cur_rho2 = init_params['cur_rho2']
    cur_rho3 = init_params['cur_rho3']
    cur_rho4 = init_params['cur_rho4']

    y1 = other_params['y1']
    y2 = other_params['y2']
    y3 = other_params['y3']
    z1 = other_params['z1']
    z2 = other_params['z2']
    z3 = other_params['z3']
    z4 = other_params['z4']
    ones = other_params['ones']

    G_iters = torch.zeros_like(G)
    n,c,w,h = G.shape
    size = end_iter - start_iter
    G_iters = G_iters.repeat(size,1,1,1)
    # print(G_iters.shape)

    for cur_iter in range(start_iter,end_iter): 
        G.requires_grad = True
        epsilon.requires_grad = False
        
        # 1.update y1 & y2
        y1 = torch.clamp((G.detach() + z1/cur_rho1), 0.0, 1.0)
        y2 = project_shifted_lp_ball(G.detach() + z2/cur_rho2, 0.5*torch.ones_like(G))      
        
        # 2.update y3
        C=G.detach()+z3/cur_rho3                       
        BC = C*B                                       
        n,c,w,h = BC.shape
        Norm = torch.norm(BC.reshape(n, c*w*h), p=2, dim=1).reshape((n,1,1,1))   
        coefficient = 1-args.lambda2/(cur_rho3*Norm)    
        coefficient = torch.clamp(coefficient, min=0)   
        BC = coefficient*BC                           
        
        y3 = torch.sum(BC, dim=0, keepdim=True)      
        

        # 3.update G
        #cnn_grad_G
        image_s = images + torch.mul(G, epsilon)
        image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
        image_s = Normalization(image_s, img_normalized_ops)

        prediction = model(image_s)
        
        if args.loss == 'ce':
            ce = nn.CrossEntropyLoss()
            loss = ce(prediction, target_label)   

        elif args.loss == 'cw':
            label_to_one_hot = torch.tensor([[target_label.item()]])
            label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
            
            real = torch.sum(prediction*label_one_hot)
            other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction - (label_one_hot*10000))
            loss = torch.clamp(other_max - real + args.confidence, min=0)
        
        
        if G.grad is not None:  #the first time there is no grad
            G.grad.data.zero_()
        loss.backward()
        cnn_grad_G = G.grad
        
        grad_G = 2*G*epsilon*epsilon*noise_Weight*noise_Weight + args.lambda1*cnn_grad_G \
                 + z1 + z2 + z3+ z4*ones + cur_rho1*(G-y1) \
                 + cur_rho2*(G-y2)+ cur_rho3*(G-y3) \
                 + cur_rho4*(G.sum().item() - args.k)*ones
                 
        G = G - cur_step*grad_G
        G = G.detach()

        G_iters[cur_iter%50] = G 



        # 4.update z1,z2,z3,z4
        z1 = z1 + cur_rho1 * (G.detach() - y1)
        z2 = z2 + cur_rho2 * (G.detach() - y2)
        z3 = z3 + cur_rho3 * (G.detach() - y3)
        z4 = z4 + cur_rho4 * (G.sum().item()-args.k)

        # 5.updating rho1, rho2, rho3, rho4
        if cur_iter % args.rho_increase_step == 0:
            cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
            cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
            cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)
            cur_rho4 = min(args.rho_increase_factor * cur_rho4, args.rho4_max)
            
        # updating learning rate
        if cur_iter % args.lr_decay_step == 0:
            cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
        
        if cur_iter % args.tick_loss_g == 0:
            cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
            noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)  
            
        cur_iter = cur_iter + 1
    
    # print(f"after iterations: {G_iters.shape}")
    G_permu = G_iters.permute(1,2,3,0)
    # print(f"after permute: {G_permu.shape}")
    # print(G_permu[0,0,0])

    init_params['cur_step_g'] = cur_step 
    init_params['cur_rho1'] = cur_rho1
    init_params['cur_rho2'] = cur_rho2
    init_params['cur_rho3'] = cur_rho3
    init_params['cur_rho4'] = cur_rho4
    other_params['y1'] = y1
    other_params['y2'] = y2
    other_params['y3'] = y3
    other_params['z1'] = z1
    other_params['z2'] = z2
    other_params['z3'] = z3
    other_params['z4'] = z4
    return init_params, other_params, G_permu 

# this is the lpbox ADMM goes. 
def update_G(model, images, target_label, epsilon, G, init_params, B, noise_Weight, out_iter, f):
    # initialize learning rate
    cur_step = init_params['cur_step_g']
    cur_rho1 = init_params['cur_rho1']
    cur_rho2 = init_params['cur_rho2']
    cur_rho3 = init_params['cur_rho3']
    cur_rho4 = init_params['cur_rho4']

    # print(f"G shape: {G.shape}")

    # initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros matrix
    y1 = torch.ones_like(G)
    y2 = torch.ones_like(G)
    y3 = torch.ones_like(G)
    z1 = torch.zeros_like(G)
    z2 = torch.zeros_like(G)
    z3 = torch.zeros_like(G)
    z4 = torch.zeros(1).cuda()
    ones = torch.ones_like(G)

    print(f"there will be {args.maxIter_g} loops for update G")
    for cur_iter in range(1,args.maxIter_g+1): 
        G.requires_grad = True
        epsilon.requires_grad = False
        
        # 1.update y1 & y2
        y1 = torch.clamp((G.detach() + z1/cur_rho1), 0.0, 1.0)
        y2 = project_shifted_lp_ball(G.detach() + z2/cur_rho2, 0.5*torch.ones_like(G))      
        
        # 2.update y3
        C=G.detach()+z3/cur_rho3                       
        BC = C*B                                       
        n,c,w,h = BC.shape
        Norm = torch.norm(BC.reshape(n, c*w*h), p=2, dim=1).reshape((n,1,1,1))   
        coefficient = 1-args.lambda2/(cur_rho3*Norm)    
        coefficient = torch.clamp(coefficient, min=0)   
        BC = coefficient*BC                           
        
        y3 = torch.sum(BC, dim=0, keepdim=True)      
        


        # 3.update G
        #cnn_grad_G
        image_s = images + torch.mul(G, epsilon)
        image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
        image_s = Normalization(image_s, img_normalized_ops)

        prediction = model(image_s)
        # print("test 111: ", image_s.shape, prediction.shape, target_label.shape)
        
        if args.loss == 'ce':
            ce = nn.CrossEntropyLoss()
            loss = ce(prediction, target_label)   

        elif args.loss == 'cw':
            label_to_one_hot = torch.tensor([[target_label.item()]]) # (1,1)
            label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda() # (1,10)

            # print("test 112: ", label_one_hot.shape, label_to_one_hot.shape)
            
            real = torch.sum(prediction*label_one_hot) 
            other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction - (label_one_hot*10000))
            loss = torch.clamp(other_max - real + args.confidence, min=0)
        
        
        if G.grad is not None:  #the first time there is no grad
            G.grad.data.zero_()
        loss.backward()
        cnn_grad_G = G.grad
        
        grad_G = 2*G*epsilon*epsilon*noise_Weight*noise_Weight + args.lambda1*cnn_grad_G \
                 + z1 + z2 + z3+ z4*ones + cur_rho1*(G-y1) \
                 + cur_rho2*(G-y2)+ cur_rho3*(G-y3) \
                 + cur_rho4*(G.sum().item() - args.k)*ones
                 
        G = G - cur_step*grad_G
        G = G.detach()

        # if f is not None:
        #     f.write(f"Iter{cur_iter},")
        #     n,c,w,h = G.shape
        #     for i in range(c):
        #         for j in range(w):
        #             for k in range(h):
        #                 f.write(f"{G[n-1,i,j,k].item()},")
        #     f.write('\n')




        # 4.update z1,z2,z3,z4
        z1 = z1 + cur_rho1 * (G.detach() - y1)
        z2 = z2 + cur_rho2 * (G.detach() - y2)
        z3 = z3 + cur_rho3 * (G.detach() - y3)
        z4 = z4 + cur_rho4 * (G.sum().item()-args.k)

        # 5.updating rho1, rho2, rho3, rho4
        if cur_iter % args.rho_increase_step == 0:
            cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
            cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
            cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)
            cur_rho4 = min(args.rho_increase_factor * cur_rho4, args.rho4_max)
            
        # updating learning rate
        if cur_iter % args.lr_decay_step == 0:
            cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
        
        if cur_iter % args.tick_loss_g == 0:
            cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
            noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)  
            
        cur_iter = cur_iter + 1

    

    res_param = {'cur_step_g': cur_step, 'cur_rho1': cur_rho1,'cur_rho2': cur_rho2, 'cur_rho3': cur_rho3,'cur_rho4': cur_rho4}
    return G, res_param

if __name__ == '__main__':
    main()

    