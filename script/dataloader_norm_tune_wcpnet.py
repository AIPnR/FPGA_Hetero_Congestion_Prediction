import os, glob
os.environ["DGLBACKEND"] = "pytorch"
import random
from datamaker import DATA_MAKER
from data_collect_wcpnet import VPR_RUN
import dgl
from dgl import load_graphs
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

RANDOM_SEED = 0

class DataLoader():
    def __init__(self, circuit_list, seed_range=200, shuffle=True, test_ratio=0.1, normalize=True,
                size=512):
        self.circuit_list = circuit_list
        self.seed_range = seed_range
        self.shuffle = shuffle
        self.test_ratio = test_ratio
        self.normalize = normalize
        self.size = size
 
    def label_normalizer(self, g):
        g.nodes['grid'].data['label'].div_(100)
        tmp = g.nodes['grid'].data['label']
        for idx in range(4):
            if torch.max(tmp[:,idx])==0:
                g.nodes['grid'].data['label'][:,idx]=torch.rand(128*128)*0.01 
                #有一些l16为全0，影响后续model train和eval，用一个接近0的随机矩阵替代
        return g.nodes['grid'].data['label'].view(128, 128, 4).permute(2,0,1)
    
                    
    def load(self):

        num_train = int(self.seed_range*(1-self.test_ratio))

        train_dataset = []
        test_dataset = []

        total = len(self.circuit_list) * self.seed_range
        pbar = tqdm(total=total, desc="Dataset loading")

        # for idx, circuit in enumerate(self.circuit_list):
        for idx, circuit in enumerate(self.circuit_list):
            # 读取 circuit下每一个graph，保存在graph_list and label_list

            label_list = []
            img_list = []
            for seed_i in range(self.seed_range):
                # rouoting_congestion_label
                data_maker = DATA_MAKER(circuit, seed_i)
                graph_path = data_maker.graph_path
                if not os.path.exists(graph_path):  
                        raise FileNotFoundError(f"No such file or directory: '{graph_path}'")
                g, g_dict = load_graphs(graph_path)
                label_list.append(self.label_normalizer(g[0]))
                # img
                vpr_run = VPR_RUN(circuit, seed_i)
                img_net = os.path.join(vpr_run.circuit_seed_wcpnet_dir, 'net.png')
                img_place = os.path.join(vpr_run.circuit_seed_wcpnet_dir, 'place.png')
                img_pin = os.path.join(vpr_run.circuit_seed_wcpnet_dir, 'pin_util.png')
                if not os.path.exists(img_net):  
                        raise FileNotFoundError(f"No such file or directory: '{img_net}'")
                if not os.path.exists(img_place):  
                        raise FileNotFoundError(f"No such file or directory: '{img_place}'")
                if not os.path.exists(img_pin):  
                        raise FileNotFoundError(f"No such file or directory: '{img_pin}'")
                img_net_tensor = torch.from_numpy(np.array(Image.open(img_net).resize((self.size, self.size)))).permute(2,0,1)
                img_place_tensor = torch.from_numpy(np.array(Image.open(img_place).resize((self.size, self.size)))).permute(2,0,1)
                img_pin_tensor = torch.from_numpy(np.array(Image.open(img_pin).convert('L').resize((self.size, self.size)))).unsqueeze(0)
                img_tensor = torch.cat([img_net_tensor, img_place_tensor, img_pin_tensor], dim=0).unsqueeze(0)
                
                img_list.append(img_tensor)                
                pbar.update() #进度条更新

            #random.seed确保在每次随机划分的traindataset和testdataset都一样
            random.seed(RANDOM_SEED + idx)
            random_index = [i for i in range(self.seed_range)]
            random.shuffle(random_index)
            for i in range(num_train):  
                train_dataset.append([img_list[random_index[i]],label_list[random_index[i]]])
            for i in range(num_train, self.seed_range):  
                test_dataset.append([img_list[random_index[i]],label_list[random_index[i]]])

        pbar.close()
        return train_dataset, test_dataset

if __name__ == '__main__':

    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../arch_blif_source/vtr7'))
    blifs_path = glob.glob(os.path.join(benchmark_path, '*.blif'))
    benchmarks_list = [os.path.basename(path).replace('.blif', '') for path in blifs_path]

    data_loader = DataLoader(circuit_list=benchmarks_list, seed_range=2)
    train_dataset, test_dataset = data_loader.load()

    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)
    pass
