import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time
from tqdm import tqdm
import os
import random
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'script'))
from dataloader_norm_tune import DataLoader
from utils import save_model_checkpoint, eval_cal, g_emb2image, BENCHMARK_LIST
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
argparser.add_argument('--device', type=str, default='0')
argparser.add_argument('--lr_rate', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--num_epoch', type=int, default=200)

argparser.add_argument('--save_model', type=bool, default=True)
argparser.add_argument('--save_model_freq', type=int, default=20)
argparser.add_argument('--eval_model_freq', type=int, default=5)

# ----- train config--------------
argparser.add_argument('--hgnn_out_dim', type=int, default=4) #每个grid node的out feture 维度
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
nn_arch=file_name
THIS_CHECKPOINT_DIR= os.path.join(CHECKPOINT_DIR, 
                nn_arch)
if not os.path.exists(THIS_CHECKPOINT_DIR): os.mkdir(THIS_CHECKPOINT_DIR)
print(THIS_CHECKPOINT_DIR)

CHECKPOINT_MODEL_DIR = os.path.join(THIS_CHECKPOINT_DIR, 'model')
if not os.path.exists(CHECKPOINT_MODEL_DIR): os.mkdir(CHECKPOINT_MODEL_DIR)
CHECKPOINT_EVAL_DIR = os.path.join(THIS_CHECKPOINT_DIR, 'eval_data')
if not os.path.exists(CHECKPOINT_EVAL_DIR): os.mkdir(CHECKPOINT_EVAL_DIR)
writer = SummaryWriter(THIS_CHECKPOINT_DIR)
#------- 全局变量定义-end---------------------------------------------



def dataset_reformulate(dataset):
    dataset_grid_feature = [g_emb2image((data[0].nodes['grid'].data['feature'])[:,[1,2,4]]) for data in dataset]
    #在这里只取h/v_net_density and rudy
    dataset_grid_label = [g_emb2image(data[0].nodes['grid'].data['label']) for data in dataset]
    return list(zip(dataset_grid_feature, dataset_grid_label))

def evaluate(test_dataset,gen,epoch):
    time_begin = time()
    eval_y_real = []
    eval_y_gen = []
    for idx, (x, y) in enumerate(test_dataset):
        x = x.half().to(DEVICE)
        y = y.half().to(DEVICE)
        gen.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                y_gen = gen(x)
                eval_y_gen.append(y_gen)
                eval_y_real.append(y)
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

    for idx, (x, y) in enumerate(loop):
        x = x.half().to(DEVICE)
        y = y.half().to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            G_loss = G_fake_loss + fn_loss(y_fake, y) * 100

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
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
    train_dataset, test_dataset = DataLoader(BENCHMARK_LIST, seed_range=args.seed_range, normalize=bool(args.dataset_norm)).load()
    train_dataset = dataset_reformulate(train_dataset)
    test_dataset = dataset_reformulate(test_dataset)
    disc = Discriminator(channels_xny=3+4, features=64, image_size=args.cnn_image_size).to(DEVICE)
    gen = Generator(in_channels=3, out_channels=4, features=16, image_size=args.cnn_image_size).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr_rate, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr_rate, betas=(0.5, 0.999))
    scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, 1, gamma=(1 - args.lr_decay))
    scheduler_gen = torch.optim.lr_scheduler.StepLR(opt_gen, 1, gamma=(1 - args.lr_decay))
    BCE = nn.BCEWithLogitsLoss()
    FN_LOSS = nn.MSELoss()


    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

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




if __name__ == "__main__":
    main()
