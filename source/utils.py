import torch
import torch.nn.functional as F
import numpy as np
import os, random, csv 
import glob

from scipy.stats import pearsonr, spearmanr, kendalltau
from skimage.metrics import structural_similarity


benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../arch_blif_source/vtr7'))
blifs_path = glob.glob(os.path.join(benchmark_path, '*.blif'))
BENCHMARK_LIST = [os.path.basename(path).replace('.blif', '') for path in blifs_path]
EVAL_DATA_SAVE_EPOCH_LIST = [10,180,200]

def save_model_checkpoint(model, optimizer, epoch, CHECKPOINT_MODEL_DIR, model_name=None):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if model_name:
        file_name = model_name+'_epoch_'+str(epoch)+'.pth'
    else: file_name = 'epoch_'+str(epoch)+'.pth'
    filename = os.path.abspath(os.path.join(CHECKPOINT_MODEL_DIR, file_name))
    torch.save(checkpoint, filename)


def load_model_checkpoint(model, optimizer, epoch, MODEL_DIR, device):
    model_path = os.path.join(MODEL_DIR, 'epoch_'+str(epoch)+'.pth')
    gen_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(gen_checkpoint["state_dict"])
    optimizer.load_state_dict(gen_checkpoint["optimizer"])



#********************************************************************************
# g_emb2image(grid_embedded)
#————————————————————————————————————————————————————————————————————————————————
# fun功能: 把graph grid的feature改为image的形式
# fun输入：pred, real
def g_emb2image(grid_embedded):
    # if grid_embedded.shape[0] % size_y != 0: raise ValueError(f"y={size_y}不能整除数组长度{grid_embedded.shape[0]}") 
    image = grid_embedded.view(-1, 128, grid_embedded.shape[1]).permute(2, 0, 1).unsqueeze(0)
    return image 



def pad_image(image, image_size):
    pad_height = image_size - image.shape[2]
    pad_width = image_size - image.shape[3]
    # 计算上下左右的填充量
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    # 使用np.pad函数进行填充
    padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
    return padded_image

def cut_image(image, cut_size):
    (pad_left, pad_right, pad_top, pad_bottom) = cut_size
    cutted_image = image[:,:, pad_top:-pad_bottom, pad_left:-pad_right]
    return cutted_image


#********************************************************************************
# eval_metric(pred, real)
# nrms, ssim, pearsonr_rho, spearmanr_rho, kendalltau_rho = eval_metric(pred, real)
#————————————————————————————————————————————————————————————————————————————————
# fun功能: 评估pred和real
# fun输入：pred, real，(type=np，shape=[100,100])
def eval_metric(pred, real):
    assert real.max()!=real.min(), 'array is contast'


    # nrms = np.sqrt(np.mean((pred - real)**2))/(np.max(real)-np.min(real))
    nrms = np.sqrt(np.mean((pred - real)**2))
    ssim = structural_similarity(pred, real, data_range=1)

    pred = pred.flatten()
    real = real.flatten()
    pearsonr_rho, _ = pearsonr(pred, real)
    spearmanr_rho, _ = spearmanr(pred, real)
    kendalltau_rho, _ = kendalltau(pred, real)
    return nrms, ssim, pearsonr_rho, spearmanr_rho, kendalltau_rho

#********************************************************************************
# eval_cal(epoch)
#————————————————————————————————————————————————————————————————————————————————
# fun功能: 读取保存在check_points/eval_dir的数据，用metric评估预测结果，并保存所有结果到csv
def eval_cal(pred_list, real_list,
             epoch, CHECKPOINT_EVAL_DIR, EVAL_BENCHMARK_LIST,
             seed_range=200, random_seed=0, test_ratio=0.1):
    
    if epoch==0:
        eval_data_path = os.path.join(CHECKPOINT_EVAL_DIR, 'eval_data_real.pth')
        torch.save(real_list, eval_data_path)
    if epoch+1 in EVAL_DATA_SAVE_EPOCH_LIST:
        eval_data_path = os.path.join(CHECKPOINT_EVAL_DIR, 'eval_data_'+str((epoch+1))+'.pth')
        torch.save(pred_list, eval_data_path)

    eval_result_path = os.path.join(CHECKPOINT_EVAL_DIR, 'eval_result_'+str((epoch+1))+'.csv')
    csvfile = open(eval_result_path, 'w')
    write = csv.writer(csvfile)
    write.writerow(['circuit_name', 
                    'nrms', 'ssim', 'pearson', 'spearman', 'kendall', 
                    'l4_x_nrms', 'l4_x_ssim', 'l4_x_pearson', 'l4_x_spearman', 'l4_x_kendall', 
                    'l4_y_nrms', 'l4_y_ssim', 'l4_y_pearson', 'l4_y_spearman', 'l4_y_kendall', 
                    'l16_x_nrms', 'l16_x_ssim', 'l16_x_pearson', 'l16_x_spearman', 'l16_x_kendall', 
                    'l16_y_nrms', 'l16_y_ssim', 'l16_y_pearson', 'l16_y_spearman', 'l16_y_kendall',
                                    ])
 
    # print('evaluating model on testdataset')
    grid_route_eval_average_list = [] #记录每个电路所有seed的平均值
    for circuit_idx, circuit in enumerate(EVAL_BENCHMARK_LIST):
        # circuit_idx, 第几个电路，总共20个

        # test dataset是随机从200个布局中选取的，在这计算他们的seed
        random.seed(random_seed+ circuit_idx)
        random_index = [i for i in range(seed_range)]
        random.shuffle(random_index)
        circuit_seed_idx = []
        for i in range(int(seed_range*(1-test_ratio)), seed_range):  
            circuit_seed_idx.append(random_index[i]) 

        # test dataset中遍历circuit每一种布局
        circuit_eval_list = [] #记录每个circuit seed的评估结果
        for i in range(int(seed_range*test_ratio)):
            test_list_idx = i + int(seed_range*test_ratio) * circuit_idx
            # 读取conges信息，从GPU中取出转为np
            pred = pred_list[test_list_idx].squeeze().cpu().numpy()#删除最前的batch_chan
            real = real_list[test_list_idx].squeeze().cpu().numpy()
            # 分别计算不同segment的metric
            circuit_eval = np.array([eval_metric(pred[i,:,:], real[i,:,:]) for i in range(pred.shape[0])])
            # 保存每一个电路的值
            circuit_eval_list.append(circuit_eval)   

        #当前电路所有seed的平均值
        circuit_eval_ave = np.mean(np.array(circuit_eval_list), axis=0)
        circuit_eval_seg_ave = np.mean(circuit_eval_ave, axis=0)
        circuit_eval_all_ave = np.vstack((circuit_eval_seg_ave.reshape(1, -1), circuit_eval_ave)).ravel(order='C')
        write.writerow([circuit, *circuit_eval_all_ave])
        grid_route_eval_average_list.append(circuit_eval_all_ave)
    #所有电路的平均值
    all_average = np.mean(np.array(grid_route_eval_average_list), axis=0)
    write.writerow(['all', 'average', *all_average])
    csvfile.close()
    print(f'\tnrms={all_average[0]:.6f} | ssim={all_average[1]:.6f} | pearson={all_average[2]:.6f} | spearman={all_average[3]:.6f} | kendall=={all_average[4]:.6f}')
    return all_average
