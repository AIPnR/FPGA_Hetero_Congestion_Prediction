import os, glob
os.environ["DGLBACKEND"] = "pytorch"
import random
from datamaker import DATA_MAKER
import dgl
from dgl import load_graphs
import torch
from tqdm import tqdm

RANDOM_SEED = 0

class DataLoader():
    def __init__(self, circuit_list, seed_range=200, shuffle=True, test_ratio=0.1, normalize=True,
                 use_grid_feature=1, use_geom_bb_edge=1, use_geom_loc_edge=1,
                 retain_net_ratio=1, retain_block_ratio=1):
        self.circuit_list = circuit_list
        self.seed_range = seed_range
        self.shuffle = shuffle
        self.test_ratio = test_ratio
        self.normalize = normalize

        self.use_grid_feature = use_grid_feature
        self.use_geom_bb_edge = use_geom_bb_edge
        self.use_geom_loc_edge = use_geom_loc_edge
        self.retain_net_ratio = retain_net_ratio
        self.retain_block_ratio = retain_block_ratio

        if self.normalize: self.normalizer_init()


    def normalizer_init(self):
        block_feature_max_list = []
        net_feature_max_list = []
        grid_feature_max_list = []
        grid_label_max_list = []
        total = len(self.circuit_list) * self.seed_range
        pbar = tqdm(total=total, desc="Dataset init")
        for circuit in self.circuit_list:
            for seed_i in range(self.seed_range):
                data_maker = DATA_MAKER(circuit, seed_i)
                graph_path = data_maker.graph_path
                if not os.path.exists(graph_path):  
                        raise FileNotFoundError(f"No such file or directory: '{graph_path}'")
                g, g_dict = load_graphs(graph_path)

                block_feature = g[0].nodes['block'].data['feature']
                block_feature_max_list.append(torch.max(block_feature, dim=0)[0])

                net_feature = g[0].nodes['net'].data['feature']
                net_feature_max_list.append(torch.max(net_feature, dim=0)[0])

                grid_feature = g[0].nodes['grid'].data['feature']
                grid_feature_max_list.append(torch.max(grid_feature, dim=0)[0])

                grid_label = g[0].nodes['grid'].data['label']
                grid_label_max_list.append(torch.max(grid_label, dim=0)[0])
                pbar.update() #进度条更新
        pbar.close() #进度条更新

        self.block_feature_max, _ = torch.max(torch.stack(block_feature_max_list), dim=0)
        self.net_feature_max, _ = torch.max(torch.stack(net_feature_max_list), dim=0)
        self.grid_feature_max, _ = torch.max(torch.stack(grid_feature_max_list), dim=0)
        self.grid_label_max, _ = torch.max(torch.stack(grid_label_max_list), dim=0)
        pass
        
    def normalizer(self, g):
        g.nodes['block'].data['feature'].div_(self.block_feature_max)
        g.nodes['net'].data['feature'].div_(self.net_feature_max)
        g.nodes['grid'].data['feature'].div_(self.grid_feature_max)
        g.nodes['grid'].data['label'].div_(100)
        tmp = g.nodes['grid'].data['label']
        for idx in range(4):
            if torch.max(tmp[:,idx])==0:
                g.nodes['grid'].data['label'][:,idx]=torch.rand(128*128)*0.01 
                #有一些l16为全0，影响后续model train和eval，用一个接近0的随机矩阵替代
        return g
    
    def unnormalizer(self, grid_label,device):
        return grid_label*self.grid_label_max.to(device)

    def DatasetTune(self, g):
        if not self.use_geom_bb_edge: #不使用gemo_bb
            g = dgl.remove_edges(g, g.edges('all', etype='geom_bb')[2], etype='geom_bb')
        
        if not self.use_geom_loc_edge: #不使用gemo_loc
            g = dgl.remove_edges(g, g.edges('all', etype='geom_loc')[2], etype='geom_loc')

        if not self.use_grid_feature: #不使用grid_feature
            g.nodes['grid'].data['feature'] = torch.zeros_like(g.nodes['grid'].data['feature'])

        if not self.retain_net_ratio == 1: #保留部分net(保留部分拓扑逻辑)
            num_nets = g.number_of_nodes('net')
            retain_net_ids = torch.randperm(num_nets)[:int(num_nets*self.retain_net_ratio)]
            g = g.subgraph({'net': retain_net_ids, \
                            'block': g.nodes('block').tolist(), \
                            'grid': g.nodes('grid').tolist()})
            
        if not self.retain_block_ratio == 1: #保留部分net(保留部分block)
            num_blocks = g.number_of_nodes('block')
            retain_block_ids = torch.randperm(num_blocks)[:int(num_blocks*self.retain_block_ratio)]
            g = g.subgraph({'net':g.nodes('net').tolist(), \
                            'block': retain_block_ids, \
                            'grid': g.nodes('grid').tolist()})
        
        return g

 
                    
    def load(self):

        num_train = int(self.seed_range*(1-self.test_ratio))

        train_dataset = []
        test_dataset = []

        total = len(self.circuit_list) * self.seed_range
        pbar = tqdm(total=total, desc="Dataset loading")

        # for idx, circuit in enumerate(self.circuit_list):
        for idx, circuit in enumerate(self.circuit_list):
            # 读取 circuit下每一个graph，保存在graph_list and label_list
            graph_list = []
            label_list = []
            for seed_i in range(self.seed_range):
                data_maker = DATA_MAKER(circuit, seed_i)
                graph_path = data_maker.graph_path
                if not os.path.exists(graph_path):  
                        raise FileNotFoundError(f"No such file or directory: '{graph_path}'")
                g, g_dict = load_graphs(graph_path)
                g = self.DatasetTune(g[0])
                if self.normalize:
                    graph_list.append(self.normalizer(g))
                else: graph_list.append(g)
                
                label_list.append(g_dict)
                pbar.update() #进度条更新

            #random.seed确保在每次随机划分的traindataset和testdataset都一样
            random.seed(RANDOM_SEED + idx)
            random_index = [i for i in range(self.seed_range)]
            random.shuffle(random_index)
            for i in range(num_train):  
                train_dataset.append([graph_list[random_index[i]],label_list[random_index[i]]])
            for i in range(num_train, self.seed_range):  
                test_dataset.append([graph_list[random_index[i]],label_list[random_index[i]]])

        pbar.close()
        return train_dataset, test_dataset

if __name__ == '__main__':

    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../arch_blif_source/titan'))
    blifs_path = glob.glob(os.path.join(benchmark_path, '*.blif'))
    benchmarks_list = [os.path.basename(path).replace('_stratixiv_arch_timing.blif', '') for path in blifs_path]

    benchmarks_list = ['sha']

    data_loader = DataLoader(circuit_list=benchmarks_list, seed_range=200, normalize=True)
    train_dataset, test_dataset = data_loader.load()

    print(train_dataset[0][0].nodes['block'].data['feature'].shape)
    print(train_dataset[0][0].nodes['grid'].data['label'].max())
    pass
