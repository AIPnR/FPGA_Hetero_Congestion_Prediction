import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from time import time
from tqdm import tqdm
import os
import numpy as np
import random
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'script'))
from dataloader_norm_tune import DataLoader
from utils import save_model_checkpoint, eval_cal, g_emb2image, BENCHMARK_LIST
from model.FHGNN import FHGNN
from model.UNET_adaptive import UNET as Generator
from model.Discriminator import Discriminator

import argparse
from torch.utils.tensorboard import SummaryWriter

argparser = argparse.ArgumentParser()

# ----- dataset config--------------
argparser.add_argument('--seed_range', type=int, default=20)
argparser.add_argument('--random_seed', type=int, default=0)
argparser.add_argument('--test_ratio', type=float, default=0.1)
argparser.add_argument('--dataset_norm', type=int, default=1) #1,0

# ----- train config--------------
argparser.add_argument('--device', type=str, default='1')
argparser.add_argument('--lr_rate', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--num_epoch', type=int, default=200)

argparser.add_argument('--save_model', type=bool, default=True)
argparser.add_argument('--save_model_freq', type=int, default=20)
argparser.add_argument('--eval_model_freq', type=int, default=5)

# ----- hetero-gnn config--------------
argparser.add_argument('--netlist_mp_type', type=str, default='sage')  #gcn,gat,sage
argparser.add_argument('--layout_mp_type', type=str, default='sage')  #gcn,gat,sage
argparser.add_argument('--space_mp_type', type=str, default='gcn')  #gcn,gat,sage
argparser.add_argument('--hagg_type', type=str, default='mean')  #mean,max,stack(nn)
argparser.add_argument('--loss_type', type=str, default='mse') #l1, mse
argparser.add_argument('--layer_num', type=int, default=2) #l1, mse
argparser.add_argument('--hidden_num', type=int, default=64) #l1, mses
# ----- hetero-gnn ablation config--------------
argparser.add_argument('--use_grid_feature', type=int, default=1) #1,0
argparser.add_argument('--use_geom_bb_edge', type=int, default=1) #1,0
argparser.add_argument('--use_geom_loc_edge', type=int, default=1) #1,0
argparser.add_argument('--retain_net_ratio', type=float, default=1) #0,0.5,1
argparser.add_argument('--retain_block_ratio', type=float, default=1) #0,0.5,1

# ----- train config--------------
argparser.add_argument('--nn_out_dim', type=int, default=4) #每个grid node的out feture 维度
argparser.add_argument('--cnn_image_size', type=int, default=128) 

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



class PREDICTOR_GEN(nn.Module):
    def __init__(self, fhgnn, gen):
        super(PREDICTOR_GEN, self).__init__()
        self.fhgnn = fhgnn
        self.gen = gen

    
    def forward(self, graph):
        _, grid_emb =  self.fhgnn(graph)

        pred_chanusage_image = self.gen(g_emb2image(grid_emb))

        real_chanusage = graph.nodes['grid'].data['label'].float()
        real_chanusage_image = g_emb2image(real_chanusage)

        return pred_chanusage_image, real_chanusage_image
    

class PREDICTOR_DISC(nn.Module):
    def __init__(self, fhgnn, disc):
        super(PREDICTOR_DISC, self).__init__()
        self.disc = disc
        self.fhgnn =fhgnn
    
    def forward(self, graph, chan_usage_image):
        _, grid_emb =  self.fhgnn(graph)
        d_out = self.disc(g_emb2image(grid_emb), chan_usage_image)
        return d_out
    
def evaluate(test_dataset,gen,epoch):
    time_begin = time()
    eval_y_real = []
    eval_y_gen = []

    loop = tqdm(test_dataset, leave=True)
    for idx, (test_data, test_data_label) in enumerate(loop):
        test_data = test_data.to(DEVICE)
        gen.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_chanusage_cut_image, real_chanusage_image = gen(test_data)
                eval_y_gen.append(pred_chanusage_cut_image)
                eval_y_real.append(real_chanusage_image)
    all_average = eval_cal(eval_y_gen, eval_y_real, 
                epoch, CHECKPOINT_EVAL_DIR, BENCHMARK_LIST, 
                args.seed_range, args.random_seed, args.test_ratio)
    
    eval_time = (time()-time_begin)/len(test_dataset)
    print('\tEvaluation: inference time (sec/img):',eval_time)
    writer.add_scalar('inference time(sec/img)/epoch', eval_time, epoch)
    global_step = (epoch+1)
    writer.add_scalar('nrms/iteration', all_average[0], global_step)
    writer.add_scalar('ssim/iteration', all_average[1], global_step)
    writer.add_scalar('pearson/iteration', all_average[2], global_step)
    writer.add_scalar('spearman/iteration', all_average[3], global_step)
    writer.add_scalar('kendall/iteration', all_average[4], global_step)


    return eval_time


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, fn_loss, bce, g_scaler, d_scaler, epoch, scheduler_disc, scheduler_gen
):
    time_begin = time()
    loop = tqdm(loader, leave=True)
    for idx, (train_data, train_data_label) in enumerate(loop):
        train_data = train_data.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            pred_chanusage_cut_image, real_chanusage_image = gen(train_data)
            D_real = disc(train_data, real_chanusage_image)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(train_data, pred_chanusage_cut_image.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        clip_grad_norm_(disc.parameters(), max_norm=1.0)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(train_data, pred_chanusage_cut_image)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            f_loss = fn_loss(pred_chanusage_cut_image, real_chanusage_image) * 100
            G_loss = G_fake_loss + f_loss 

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        clip_grad_norm_(gen.parameters(), max_norm=1.0)  # 添加这一行来裁剪梯度
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                epoch = epoch,
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_loss=format(D_loss,'.3f'),
                G_loss=format(G_loss,'.3f'),
            )
    scheduler_disc.step()
    scheduler_gen.step()

    train_time = (time()-time_begin)
    writer.add_scalar('training time/epoch', train_time, epoch)
    return train_time

def main():
    if args.loss_type=='l1':
        loss_f = nn.L1Loss()
    elif args.loss_type=='mse':
        loss_f = nn.MSELoss()

    dataloader = DataLoader(BENCHMARK_LIST, seed_range=args.seed_range, normalize=bool(args.dataset_norm))
    train_dataset, test_dataset = dataloader.load()

    in_block_feats = train_dataset[0][0].nodes['block'].data['feature'].shape[1]
    in_net_feats = train_dataset[0][0].nodes['net'].data['feature'].shape[1]
    in_grid_feats = train_dataset[0][0].nodes['grid'].data['feature'].shape[1]

    fhgnn_gen = FHGNN(in_block_feats, in_net_feats, in_grid_feats,
                  # super_parameter
                  hidden_feats = args.hidden_num, 
                  layer_num = args.layer_num,  
                  target_num= args.nn_out_dim,
                  #hetero_NN arch
                  netlist_mp_type = args.netlist_mp_type,
                  layout_mp_type = args.layout_mp_type,
                  space_mp_type = args.space_mp_type,
                  hagg_type = args.hagg_type,
                  use_grid_feature = args.use_grid_feature,
                  device = DEVICE
                  ).to(DEVICE)
    gen_base = Generator(in_channels=args.hidden_num, out_channels=args.nn_out_dim, 
                    features=args.hidden_num, 
                    image_size=args.cnn_image_size).to(DEVICE)  
    gen  = PREDICTOR_GEN(fhgnn_gen, gen_base)
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr_rate, betas=(0.5, 0.999))
    scheduler_gen = torch.optim.lr_scheduler.StepLR(opt_gen, 1, gamma=(1 - args.lr_decay))
    g_scaler = torch.cuda.amp.GradScaler()

    fhgnn_disc = FHGNN(in_block_feats, in_net_feats, in_grid_feats,
                # super_parameter
                hidden_feats = args.nn_out_dim, 
                layer_num = args.layer_num,  
                target_num= args.nn_out_dim,
                #hetero_NN arch
                netlist_mp_type = args.netlist_mp_type,
                layout_mp_type = args.layout_mp_type,
                space_mp_type = args.space_mp_type,
                hagg_type = args.hagg_type,
                use_grid_feature = args.use_grid_feature,
                device = DEVICE
                ).to(DEVICE)
    disc_base = Discriminator(channels_xny=args.nn_out_dim*2, 
                         features=args.hidden_num, 
                         image_size=args.cnn_image_size).to(DEVICE)
    disc = PREDICTOR_DISC(fhgnn_disc, disc_base)
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr_rate, betas=(0.5, 0.999))
    scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, 1, gamma=(1 - args.lr_decay))
    d_scaler = torch.cuda.amp.GradScaler()

    BCE = nn.BCEWithLogitsLoss()
    FN_LOSS = nn.MSELoss()

    train_time_list = []
    eval_time_list = []
    
    for epoch in range(args.num_epoch):
        current_lr = opt_gen.param_groups[0]['lr']
        print(f'##### EPOCH {epoch+1}, Learning rate: {current_lr}')
        train_time = train_fn(disc, gen, train_dataset, opt_disc, opt_gen, FN_LOSS, BCE, g_scaler, d_scaler, epoch, scheduler_disc, scheduler_gen)
        train_time_list.append(train_time)
        if ((epoch+1)%args.eval_model_freq ==0) or epoch==0 :
            eval_time = evaluate(test_dataset, gen, epoch)
            eval_time_list.append(eval_time)
        if args.save_model and ((epoch+1)%args.save_model_freq ==0):
            save_model_checkpoint(gen, opt_gen, epoch+1, CHECKPOINT_MODEL_DIR, model_name = 'gen')
            save_model_checkpoint(disc, opt_disc, epoch+1, CHECKPOINT_MODEL_DIR, model_name = 'disc')




    average_train_time = np.array(train_time_list).mean()
    print(f"\tAverage training time: {format(average_train_time,'.3f')}")
    average_eval_time = np.array(eval_time_list).mean()
    print(f"\tAverage inference time (sec/img): {format(average_eval_time, '.3f')}")
    writer.add_text("Average training time:", f"{format(average_train_time,'.3f')}")
    writer.add_text("Average inference time (sec/img):", f"{format(average_eval_time, '.3f')}")

if __name__ == '__main__':
    main()
