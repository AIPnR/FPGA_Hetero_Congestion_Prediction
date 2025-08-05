import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn
import dgl.data

class Discriminator(nn.Module):
    def __init__(self, num_features_in, num_features_out):
        super(Discriminator, self).__init__()
        self.conv1 = dglnn.GraphConv(num_features_in, 128)  # 输入特征数为 num_features * 2
        self.conv2 = dglnn.GraphConv(128, 128)  # 
        self.conv3 = dglnn.GraphConv(128, 128)  # 
        self.conv4 = dglnn.GraphConv(128, num_features_out)  # 输出特征数为 4

    def forward(self, g, in_feat):
        h = torch.relu(self.conv1(g, in_feat))
        h = torch.relu(self.conv2(g, h))
        h = torch.relu(self.conv3(g, h))
        h = self.conv4(g, h)
        return h


def disc_in_maker(x_graph, y_graph):
    grid = x_graph.nodes('grid')
    grid_edges = x_graph.edges(etype='grid_edge')

    # 创建同构图
    homograph = dgl.graph((grid_edges[0], grid_edges[1]), num_nodes=len(grid))
    feature = x_graph.nodes['grid'].data['feature']
    label = y_graph.nodes['grid'].data['label']
    homograph.ndata['feature'] = torch.cat((feature, label),dim=1).float()
    return homograph, homograph.ndata['feature']

def test():
    from dgl import load_graphs

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    graph_path = '/home/mavo/AA_RL_Placer/hetero_gnn_routability_titan/dataset/neuron/neuron_seed0_dgl_graph.bin'
    g, g_dict = load_graphs(graph_path)

    disc_in_graph, disc_in_graph_feature = disc_in_maker(g[0], g[0])

    in_block_feats = disc_in_graph_feature.shape[1]

    model = Discriminator(in_block_feats, 4)
 
    pred = model( disc_in_graph, disc_in_graph_feature)
    pass



if __name__ == "__main__":
    test()
