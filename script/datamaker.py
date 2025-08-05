#********************************************************************************
# dataset_maker
#————————————————————————————————————————————————————————————————————————————————
# 功能描述：
#   1. class DATA_MAKER，调用VPR_RUN, 处理vpr out为制作heteroDGLGRAPH的要素(edge, feature)
#       a. 由3种nodes组成：网表中的net，网表中的blocks，layout中的grid
#       b. edge分别为: net-topo-blocks(input)， block-topo-net(output)
#                     net-geom_bb-grid, block-gemo_loc-grid
#                     grid-grid_edge-grid
#       c. featuraztion:
#           a). net_feature: bb_high, bb_length, bb_area, block(nodes)_density
#           b), block_feature: type_io, type_clb, num_in_pin, num_out_pin
#           c), grid_feature: type_io, type_clb, 
#                             horizontal/vertical net density, rudy, pin_density
#                             label_route_denmand, label_switch_demand (此两项是再表示到grid_level)
#       d. label: chanx_usage. chany_usage, switch_cb_x, switch_cb_y, switch_sb
#          注意：routing_usage=track_num/100
#   2. class RougesDataset(Dataset) 
#      a. 制作数据集:调用DATA_MAKER生成的数据制作hetero-graph dataset,保存在DATA_DIR,
#         同一circuit的不同布局(seed)作为不同graph，同一circuit放在一个graph set中
#      b. 调用数据集
#         rouge_dataset = RougesDataset(circuit_name,fpga_size=100, io_cap=8, seed_range=200, force_save = True)
#         graph list: rouge_dataset.g_list
#         graph label list : rouge_dataset.label_list
#         graph, graph_label = rouge_dataset[i]
#————————————————————————————————————————————————————————————————————————————————


import os, glob
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl import save_graphs
import torch
import torch.nn as nn
import tqdm
import numpy as np
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

from npr_reader import VPR_READER

DATA_DIR = '../z_vtr7_dataset'

class DATA_MAKER():
    def __init__(self, circuit_name, seed=0, force_save=False):
        self.circuit_name = circuit_name            #circuit name
        self.seed = seed                            #place seed
        self.force_save = force_save                #是否强制保存
        
        # 地址相关
        script_dir = os.path.dirname(__file__)
        dataset_dir = os.path.abspath(os.path.join(script_dir, DATA_DIR))
        if not os.path.exists(dataset_dir): os.mkdir(dataset_dir)
        dataset_circuit_dir = os.path.abspath(os.path.join(dataset_dir, circuit_name))
        if not os.path.exists(dataset_circuit_dir ): os.mkdir(dataset_circuit_dir )
        self.graph_path = os.path.join(dataset_circuit_dir, \
            self.circuit_name+'_seed'+str(self.seed)+'_dgl_graph.bin')

         
    #--------------------------------------------------------------
    def q_factor(self, num_terminal):  
        q_constant = [
            1.0,    1.0,    1.0,    1.0828, 1.1536, 1.2206, 1.2823, 1.3385, 1.3991, 1.4493,
            1.4974, 1.5455, 1.5937, 1.6418, 1.6899, 1.7304, 1.7709, 1.8114, 1.8519, 1.8924,
            1.9288, 1.9652, 2.0015, 2.0379, 2.0743, 2.1061, 2.1379, 2.1698, 2.2016, 2.2334,
            2.2646, 2.2958, 2.3271, 2.3583, 2.3895, 2.4187, 2.4479, 2.4772, 2.5064, 2.5356,
            2.5610, 2.5864, 2.6117, 2.6371, 2.6625, 2.6887, 2.7148, 2.7410, 2.7671, 2.7933]
        if (num_terminal > 50):
            q = 2.7933 + 0.02616 * (num_terminal - 50)
        else:
            q = q_constant[num_terminal-1]
        return(q)

    def grid_maker_hetero(self):
    #********************************************************************************
    # grid_edge_maker
    # src, dst = grid_edge_maker
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 为所有在几何空间(上下左右)相邻的grid间建立双向edge
    #fun输出: grid edge的src/dst grid id 
        src = []
        dst = []
        #添加纵向adjecnt
        for row in range(self.fpga_size_x):
            for col in range(self.fpga_size_y-1):
                src.append(self.grid_matrix_idx[row][col])
                dst.append(self.grid_matrix_idx[row][col+1])
        #添加横向adjecnt
        for row in range(self.fpga_size_x-1):
            for col in range(self.fpga_size_y):
                src.append(self.grid_matrix_idx[row][col])
                dst.append(self.grid_matrix_idx[row+1][col])

        #添加Macro adjecnt
        macro_type_height_list = [(key, value['height']) for key, value in self.block_info_list.items() if value['height'] > 1]
        for macro_type, macro_height in macro_type_height_list:
            rows, cols = np.where(self.grid_matrix_type == macro_type) 
            for row in np.unique(rows):
                if len(cols) % macro_height != 0: raise ValueError("The length of cols cannot be evenly divided by height")
                split_cols = np.array_split(cols, len(cols) / macro_height)
                comb_list = [list(combinations(sub_arr, 2)) for sub_arr in split_cols]
                flat_comb_list = [item for sublist in comb_list for item in sublist]
                for pair in flat_comb_list:
                    if abs(pair[0] - pair[1]) > 1:
                        src.append(self.grid_matrix_idx[row][pair[0]])
                        dst.append(self.grid_matrix_idx[row][pair[1]])

        # macro的type
        grid_type_embed = torch.zeros(self.fpga_size_x * self.fpga_size_y,1)
        for row in range(self.fpga_size_x):
            for col in range(self.fpga_size_y):
                grid_type = self.grid_matrix_type[row][col]
                grid_type_embed[self.grid_matrix_idx[row][col]] = self.type_embedding(torch.tensor(self.block_info_list[grid_type]['type_id']))


        bi_src = src + dst
        bi_dst = dst + src
        return (bi_src, bi_dst), grid_type_embed

    def topo_maker(self):
    #********************************************************************************
    # topo_maker
    # (net2block_src, net2block_dst), (block2net_src, block2net_dst) = geom_loc_macker
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 生成有向的net与block间的edge
    #fun输入: self.block_list，nodes_list:[ndoes_id，nodes_name, nodes_type, inputs_netname, outputs_netname]
    #        self.nets_id，nets_id = {net_name: ID, }
    #fun输出: net2block的src和dst，block2net的src和dst
        # net -> block (input)
        net2block_src = []
        net2block_dst = []
        # block -> net (output)
        block2net_src = []
        block2net_dst = []

        for block in self.block_list:
            # net -> block (input)
            for in_net in block[3]:
                net2block_src.append(self.nets_id[in_net])             
                net2block_dst.append(block[0]) 
            # block -> net (output)
            for out_net in block[4]:          
                block2net_src.append(block[0]) 
                block2net_dst.append(self.nets_id[out_net])  

        return (net2block_src, net2block_dst), (block2net_src, block2net_dst)

    def geom_loc_maker(self):
    #********************************************************************************
    # geom_loc_macker
    # src, dst = geom_loc_macker
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 为每一个block指向layout中特定的grid
    #fun输入: self.block_location_list：[ndoes_id, nodes_name, nodes_place(x,y), node_subblk]
    #fun输出: geom_loc的src(block)/dst(grid) id 需注意macro block只连接到最下坐标
        src= []
        dst= []
        for block in self.block_location_list:
            src.append(block[0]) 
            row, col = block[2]
            dst.append(self.grid_matrix_idx[row][col])
        return src, dst

    def bb_calculator(self):
    #********************************************************************************
    # bb_calculator
    # src, dst, net_feature[] = bb_macker
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: net指向bb内所有的grid, 计算net的feature
    #fun输入: self.nets_list，nets_list(dict):['net_name': [nodes_id,...]]
    #        self.block_location_list 
    #fun输出: [0] net-geom-grid, net作为src，grid的编号作为dst, 需要注意net连接到所覆盖的所有grid
    #        [1] net feature：bb_high, bb_length, bb_area, bb_block_density
    #        [2] grid_feature: grid_h_net_density, grid_v_net_density, grid_rudy
        src = []
        dst = []
        net_num = len(self.nets_id)
        bb_high = torch.zeros(net_num ,1)
        bb_length = torch.zeros(net_num ,1)
        bb_area = torch.zeros(net_num ,1)
        bb_block_density = torch.zeros(net_num ,1)

        #-----------for grid feature----------------
        grid_v_net_density = torch.zeros(self.fpga_size_x * self.fpga_size_y,1)
        grid_h_net_density = torch.zeros(self.fpga_size_x * self.fpga_size_y,1)
        grid_rudy          = torch.zeros(self.fpga_size_x * self.fpga_size_y,1)
        
        print(f'{self.circuit_name}_seed{self.seed} bound box calculating')
        for i, (net_name, nodes_id) in \
            tqdm.tqdm(enumerate(self.nets_list.items()), total=len(self.nets_list.keys())):
        #遍历每一个net
            net_row = []
            net_col = []
            for node_id in nodes_id:
                net_row.append(self.block_location_list[node_id][2][0])
                col = self.block_location_list[node_id][2][1]
                # 以下因为dsp等macro竖向占位多个
                height = self.block_info_list[self.block_list[node_id][2]]['height']
                for col_offset in range(height):
                    net_col.append(col+col_offset)
            #net中每一个block的横纵坐标分别存于net_row and net_col

            #net feature
            high = max(net_row)+1 - min(net_row)
            length = max(net_col)+1 - min(net_col)
            bb_high[self.nets_id[net_name]] = high
            bb_length[self.nets_id[net_name]] = length
            bb_area[self.nets_id[net_name]] = high*length
            bb_block_density[self.nets_id[net_name]] = len(nodes_id)/(high*length)

            # 遍历bb中的每一个blcok坐标，
            for row in range(min(net_row),max(net_row)+1):
                for col in range(min(net_col),max(net_col)+1):
                    #转为grid_id作为dst
                    #edge: gemo_bb
                    src.append(self.nets_id[net_name])
                    dst.append(self.grid_matrix_idx[row][col])
                    #-----------for grid feature----------------
                    grid_h_net_density[self.grid_matrix_idx[row][col]] += 1/high
                    grid_v_net_density[self.grid_matrix_idx[row][col]] += 1/length
                    grid_rudy[self.grid_matrix_idx[row][col]] += self.q_factor(len(nodes_id))*(high+length)/(high*length)
        return (src, dst), \
                torch.cat((bb_high, bb_length, bb_area, bb_block_density), dim=1),\
                    torch.cat((grid_h_net_density, grid_v_net_density, grid_rudy),dim=1)
                #gemo_net_grid, net feature, grid_feature

    def block_feature_maker(self):
    #********************************************************************************
    # block_feature_maker
    # type_io, type_clb, num_in_pin, num_out_pin = grid_edge_maker
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 提取block的特征, 为grid制作pin_density
    #fun输出: type_io, type_clb, num_in_pin, num_out_pin
        num_block = len(self.block_list)
        type_embed = torch.zeros(num_block,1)
        bloock_height = torch.zeros(num_block,1)
        num_in_pin = torch.zeros(num_block,1)
        num_out_pin = torch.zeros(num_block,1)

        pin_density = torch.zeros(self.fpga_size_x * self.fpga_size_y,1)

        for i in range(num_block):
            block_type = self.block_list[i][2]
            type_embed[i] = self.type_embedding(torch.tensor(self.block_info_list[block_type]['type_id']))
            bloock_height[i] = self.block_info_list[block_type]['height']
            num_in_pin[i] = len(self.block_list[i][3])
            num_out_pin[i] = len(self.block_list[i][4])
            block_coord_x, block_coord_y = self.block_location_list[i][2]
            pin_density[self.grid_matrix_idx[block_coord_x][block_coord_y]] = num_in_pin[i]+num_out_pin[i]
        return torch.cat((type_embed, bloock_height, num_in_pin, num_out_pin),dim=1), pin_density
        
    def routing_demand_maker(self):
    #********************************************************************************
    # grid_routing_demand_maker
    # grid_type_io, grid_type_clb, grid_routing_demand = grid_routing_demand_maker
    #————————————————————————————————————————————————————————————————————————————————
    #fun功能: 把VPR中读取的routing chanx, chany usage 平均到每一个block
    #        平均方法就是求block四周chan usage的平均数
    #fun输入: route_conges[0]:chanx_usage, [1]:chany_usage
    #fun输出: grid_type_io, grid_type_clb
    #        grid_routing_demand[row][col]  

        # chanx_usage_l4, chany_usage_l4, chanx_usage_l16, chany_usage_l16  = self.chan_usage
        chan_usage_flatten = [torch.from_numpy(t).view(-1, 1) for t in self.chan_usage]   
        chan_usage_concatenated = torch.cat(chan_usage_flatten, dim=1)
        return chan_usage_concatenated

    def graph_maker(self):        
        if not os.path.exists(self.graph_path) or self.force_save:
            
            # 初始化VPR_READER, 读取相关参数------------------------
            vpr_reader = VPR_READER(self.circuit_name, self.seed)

            self.block_location_list = vpr_reader.place_list
            self.nets_id = vpr_reader.nets_id
            self.block_list = vpr_reader.nodes_list
            self.nets_list = vpr_reader.nets_list

            self.block_info_list = vpr_reader.block_info_list
            self.grid_matrix_type = vpr_reader.matrix

            #grid和node type从one_hot编码换为nn.Embedding编码     
            torch.manual_seed(0)
            self.type_embedding = nn.Embedding(len(self.block_info_list), 1)

            self.fpga_size_x, self.fpga_size_y = vpr_reader.fpga_size
            self.grid_matrix_idx = torch.arange(0, self.fpga_size_x * self.fpga_size_y)\
                .view(self.fpga_size_x, self.fpga_size_y)              #定义grid矩阵，方便coord2id的转换

            self.chan_usage = vpr_reader.chan_usage
            # chanx_usage_l4, chany_usage_l4, chanx_usage_l16, chany_usage_l16 
            self.critical_path = torch.FloatTensor([vpr_reader.critical_path])
            self.wl = torch.FloatTensor([vpr_reader.wl])
            #-----------------------------------------------------------

            # 按照关参数计算graph edge and feature------------------------
            grid_edge, grid_type_embed = self.grid_maker_hetero()
            topo_in, topo_out = self.topo_maker()
            geom_loc = self.geom_loc_maker()
            geom_bb, net_feature, grid_feature = self.bb_calculator()
            block_feature, pin_density = self.block_feature_maker()
            routing_demand = self.routing_demand_maker()
            #-----------------------------------------------------------

            # 制作graph 并保存--------------------------------------------
            g = dgl.heterograph({
                ('grid','grid_edge','grid'):(grid_edge),
                ('net','topo_in','block'):(topo_in),
                ('block','topo_out','net'):(topo_out),
                ('block','geom_loc','grid'):(geom_loc),
                ('net','geom_bb','grid'):(geom_bb),
                })
            g.nodes['block'].data['feature'] = block_feature
            g.nodes['net'].data['feature'] = net_feature
            g.nodes['grid'].data['feature'] = \
                torch.cat((grid_type_embed, grid_feature, pin_density),dim=1)
            # grid_type, grid_h_net_density, grid_v_net_density, grid_rudy, pin_density
            g.nodes['grid'].data['label'] = routing_demand
            print('graph_saving... ', self.circuit_name+'_seed'+str(self.seed))     
            save_graphs(self.graph_path, g, {'cp':self.critical_path, 'wl':self.wl})
        
        else: print('graph already exists...', self.circuit_name+'_seed'+str(self.seed))


if __name__ == '__main__':

    benchmark_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../arch_blif_source/vtr7'))
    blifs_path = glob.glob(os.path.join(benchmark_path, '*.blif'))
    benchmarks_list = [os.path.basename(path).replace('.blif', '') for path in blifs_path]



    def process_data_maker(circuit_name, seed_i):
        data_maker = DATA_MAKER(circuit_name, seed=seed_i)
        data_maker.graph_maker()


    # 创建一个包含25个进程的进程池
    with ProcessPoolExecutor(max_workers=25) as executor:
        # 为每个circuit_name和seed_i创建一个任务
        for circuit_name in benchmarks_list:
            for seed_i in range(200):
                executor.submit(process_data_maker, circuit_name, seed_i)



