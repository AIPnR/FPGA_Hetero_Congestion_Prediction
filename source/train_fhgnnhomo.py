import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
from tqdm import tqdm
import os
import numpy as np
import random
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'script'))
from dataloader_norm_tune import DataLoader
from utils import save_model_checkpoint, eval_cal, g_emb2image, BENCHMARK_LIST
from model.FHGNN_homo import FHGNN

import argparse
from torch.utils.tensorboard import SummaryWriter


argparser = argparse.ArgumentParser()

# ----- dataset config--------------
argparser.add_argument('--seed_range', type=int, default=20)
argparser.add_argument('--random_seed', type=int, default=0)
argparser.add_argument('--test_ratio', type=float, default=0.1)
argparser.add_argument('--dataset_norm', type=int, default=1) #1,0

# ----- train config--------------
argparser.add_argument('--device', type=str, default='0')
argparser.add_argument('--lr_rate', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--num_epoch', type=int, default=200)

argparser.add_argument('--save_model', type=bool, default=True)
argparser.add_argument('--save_model_freq', type=int, default=20)
argparser.add_argument('--eval_model_freq', type=int, default=5)

# ----- hetero-gnn config--------------
argparser.add_argument('--netlist_mp_type', type=str, default='sage')  #gcn,gat,sage
argparser.add_argument('--layout_mp_type', type=str, default='sage')  #gcn,gat,sage
argparser.add_argument('--space_mp_type', type=str, default='sage')  #gcn,gat,sage
argparser.add_argument('--hagg_type', type=str, default='mean')  #mean,max,stack(nn)
argparser.add_argument('--loss_type', type=str, default='mse') #l1, mse
argparser.add_argument('--layer_num', type=int, default=2) #l1, mse
argparser.add_argument('--hidden_num', type=int, default=64) #l1, mse
# ----- hetero-gnn ablation config--------------
argparser.add_argument('--use_grid_feature', type=int, default=0) #1,0
argparser.add_argument('--use_geom_bb_edge', type=int, default=0) #1,0
argparser.add_argument('--use_geom_loc_edge', type=int, default=1) #1,0
argparser.add_argument('--retain_net_ratio', type=float, default=1) #0,0.5,1
argparser.add_argument('--retain_block_ratio', type=float, default=1) #0,0.5,1

# ----- train config--------------
argparser.add_argument('--out_dim', type=int, default=4) #每个grid node的out feture 维度

args = argparser.parse_args()

#------- 全局变量定义----------------------------------------------

    
if args.device == '0':
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args.device == '1':
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

    

CHECKPOINT_DIR= os.path.abspath(os.path.join(os.path.dirname(__file__), './zheck_points'))
if not os.path.exists(CHECKPOINT_DIR): os.mkdir(CHECKPOINT_DIR)                               

file_name =  os.path.basename(os.path.abspath(__file__)).replace('train_', '').replace('.py', '')
nn_arch=file_name + '_' + args.netlist_mp_type+args.layout_mp_type+args.space_mp_type+args.hagg_type
THIS_CHECKPOINT_DIR= os.path.join(CHECKPOINT_DIR, 
                nn_arch+'_layer'+str(args.layer_num)+'_hidden'+str(args.hidden_num)+\
    '_dnorm'+str(args.dataset_norm)+         
    '_gridf'+str(args.use_grid_feature)+'_gbe'+str(args.use_geom_bb_edge)+'_gle'+str(args.use_geom_loc_edge)+\
                        '_net'+str(args.retain_net_ratio)+'_block'+str(args.retain_block_ratio))
if not os.path.exists(THIS_CHECKPOINT_DIR): os.mkdir(THIS_CHECKPOINT_DIR)
print(THIS_CHECKPOINT_DIR)

CHECKPOINT_MODEL_DIR = os.path.join(THIS_CHECKPOINT_DIR, 'model')
if not os.path.exists(CHECKPOINT_MODEL_DIR): os.mkdir(CHECKPOINT_MODEL_DIR)
CHECKPOINT_EVAL_DIR = os.path.join(THIS_CHECKPOINT_DIR, 'eval_data')
if not os.path.exists(CHECKPOINT_EVAL_DIR): os.mkdir(CHECKPOINT_EVAL_DIR)
writer = SummaryWriter(THIS_CHECKPOINT_DIR)
#------- 全局变量定义-end---------------------------------------------

def forward_process(model, data):
    pred, _  = model.forward(data)
    pred = g_emb2image(pred)
    with torch.no_grad():
        real = data.nodes['grid'].data['label'].float()
        real = g_emb2image(real)
    return pred, real

def main():
    if args.loss_type=='l1':
        loss_f = nn.L1Loss()
    elif args.loss_type=='mse':
        loss_f = nn.MSELoss()

    dataloader = DataLoader(BENCHMARK_LIST, seed_range=args.seed_range , normalize=bool(args.dataset_norm),
                            use_grid_feature = args.use_grid_feature, 
                            use_geom_bb_edge = args.use_geom_bb_edge, 
                            use_geom_loc_edge= args.use_geom_loc_edge,
                            retain_net_ratio = args.retain_net_ratio, 
                            retain_block_ratio=args.retain_block_ratio)
    train_dataset, test_dataset = dataloader.load()

    in_block_feats = train_dataset[0][0].nodes['block'].data['feature'].shape[1]
    in_net_feats = train_dataset[0][0].nodes['net'].data['feature'].shape[1]
    in_grid_feats = train_dataset[0][0].nodes['grid'].data['feature'].shape[1]

    model = FHGNN(in_block_feats, in_net_feats, in_grid_feats,
                  # super_parameter
                  hidden_feats = args.hidden_num, 
                  layer_num = args.layer_num,  
                  target_num= args.out_dim,
                  #hetero_NN arch
                  netlist_mp_type = args.netlist_mp_type,
                  layout_mp_type = args.layout_mp_type,
                  space_mp_type = args.space_mp_type,
                  hagg_type = args.hagg_type,
                  use_grid_feature = args.use_grid_feature,
                  device = DEVICE
                  ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

    # global_step = 0
    for epoch in range(args.num_epoch):
        print(f'##### EPOCH {epoch+1}, Learning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        train_time_list = []
        def train(train_dataset):
            global_step = epoch*(len(train_dataset))
            model.train()
            time_begin = time()

            random.shuffle(train_dataset)

            average_loss = 0
            loop = tqdm(train_dataset, leave=True)

            
            for idx, (train_data, train_data_label) in enumerate(loop):
                
                global_step = global_step+1

                train_data = train_data.to(DEVICE)
                optimizer.zero_grad()

                # a = torch.cuda.memory_allocated()
                # print("GPU memory:{}".format(torch.cuda.memory_allocated()-a))

                # pred, real = forward_process(model, train_data)
                pred, _  = model.forward(train_data)
                with torch.no_grad():
                    real = train_data.nodes['grid'].data['label'].float()

                loss = loss_f(pred, real)
                

                loss.backward()
                optimizer.step()
                

                writer.add_scalar('loss/iteration', loss, global_step)
                if idx % 10 == 0:
                    loop.set_postfix(
                        epoch = epoch,
                        loss=format(loss,'.3f'))
                    average_loss += loss
                
                torch.cuda.empty_cache()
                

            scheduler.step()
            train_time = (time()-time_begin)
            train_time_list.append(train_time)
            print(f"\tEpoch: {epoch+1}, training time: {format(train_time,'.3f')}, average loss: {format(average_loss*10/(idx+1),'.3f')}")
            writer.add_scalar('training time/epoch', train_time, epoch)

        eval_time_list = []
        def evaluate(test_dataset):
            model.eval()
            # print(f'#####Evaluate:')
            pred_list = []
            real_list = []
            time_begin = time()
            with torch.no_grad():
                for j, (test_data, test_data_label) in enumerate(test_dataset):
                    test_data = test_data.to(DEVICE)
                    optimizer.zero_grad()
                    pred, real = forward_process(model, test_data)
                    pred_list.append(pred)
                    real_list.append(real)

            eval_time = (time()-time_begin)/len(test_dataset)
            eval_time_list.append(eval_time)
            print('\tEvaluation: inference time (sec/img):',eval_time)
            writer.add_scalar('inference time(sec/img)/epoch', eval_time, epoch)

            # eval metric计算
            all_average = eval_cal(pred_list, real_list, 
                                   epoch, CHECKPOINT_EVAL_DIR, BENCHMARK_LIST, 
                                   args.seed_range, args.random_seed, args.test_ratio)
            global_step = (epoch+1)
            writer.add_scalar('nrms/iteration', all_average[0], global_step)
            writer.add_scalar('ssim/iteration', all_average[1], global_step)
            writer.add_scalar('pearson/iteration', all_average[2], global_step)
            writer.add_scalar('spearman/iteration', all_average[3], global_step)
            writer.add_scalar('kendall/iteration', all_average[4], global_step)


        # print("GPU memory:{}".format(torch.cuda.memory_allocated()))
        train(train_dataset)
        if args.save_model and ((epoch+1)%args.save_model_freq ==0):
            save_model_checkpoint(model, optimizer, epoch+1, CHECKPOINT_MODEL_DIR)

        if ((epoch+1)%args.eval_model_freq ==0) or epoch==0 :
            evaluate(test_dataset)



    average_train_time = np.array(train_time_list).mean()
    print(f"\tAverage training time: {format(average_train_time,'.3f')}")
    average_eval_time = np.array(eval_time_list).mean()
    print(f"\tAverage inference time (sec/img): {format(average_eval_time, '.3f')}")
    writer.add_text("Average training time:", f"{format(average_train_time,'.3f')}")
    writer.add_text("Average inference time (sec/img):", f"{format(average_eval_time, '.3f')}")

if __name__ == '__main__':
    main()
