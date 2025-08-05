import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn
import dgl.data

class FHGNN(nn.Module):
    def __init__(self, in_block_feats, in_net_feats, in_grid_feats,
                  hidden_feats = 32, 
                  target_num = 4,
                  layer_num = 3,  
                  #hetero_NN arch
                  netlist_mp_type = 'gcn',
                  layout_mp_type = 'gcn',
                  space_mp_type = 'gcn',
                  hagg_type = 'sum',
                  use_grid_feature = True,
                  device = 'cuda:0'
                  ):  #hetero_graph不同类型node的聚合方式
                                         
        super().__init__()
        gat_n_heads = 4
        self.target_num = int(target_num)
        self.layer_num = int(layer_num)
        self.netlist_mp_type = netlist_mp_type
        self.layout_mp_type = layout_mp_type
        self.space_mp_type = space_mp_type
        self.hagg_type = hagg_type
        self.hidden_feats = hidden_feats
        self.use_grid_feature = use_grid_feature
        self.device = device



        #feature initial 
        self.lin_block = nn.Linear(in_block_feats, hidden_feats//2)
        self.lin_net   = nn.Linear(in_net_feats,   hidden_feats//2)
        self.lin_grid  = nn.Linear(in_grid_feats,  hidden_feats//2)

        self.lin_block_2 = nn.Linear(hidden_feats//2, hidden_feats)
        self.lin_net_2   = nn.Linear(hidden_feats//2,   hidden_feats)
        self.lin_grid_2  = nn.Linear(hidden_feats//2,  hidden_feats)

        self.lin_grid_agg = nn.Linear(hidden_feats*3, hidden_feats)
        self.lin_grid_agg2 = nn.Linear(hidden_feats, hidden_feats)

             
        self.gnn_layers = nn.ModuleList(
            [dglnn.HeteroGraphConv({
                'topo_out':
                    dglnn.GraphConv(hidden_feats, hidden_feats)
                    if netlist_mp_type == 'gcn' else
                    dglnn.GATConv(hidden_feats, hidden_feats//gat_n_heads, gat_n_heads)
                    if netlist_mp_type == 'gat' else
                    dglnn.SAGEConv(hidden_feats, hidden_feats, 'pool'),
                'topo_in':
                    dglnn.GraphConv(hidden_feats, hidden_feats)
                    if netlist_mp_type == 'gcn' else
                    dglnn.GATConv(hidden_feats, hidden_feats//gat_n_heads, gat_n_heads)
                    if netlist_mp_type == 'gat' else
                    dglnn.SAGEConv(hidden_feats, hidden_feats, 'pool'),
                    # if topo_conv_type == 'SAGE' else
                'grid_edge':
                    dglnn.GraphConv(hidden_feats, hidden_feats)
                    if layout_mp_type == 'gcn' else
                    dglnn.GATConv(hidden_feats, hidden_feats//gat_n_heads, gat_n_heads)
                    if layout_mp_type == 'gat' else
                    dglnn.SAGEConv(hidden_feats, hidden_feats, 'pool'),
                    # if topo_conv_type == 'SAGE' else
                'geom_loc':
                    dglnn.GraphConv(hidden_feats, hidden_feats)
                    if space_mp_type == 'gcn' else
                    dglnn.GATConv(hidden_feats, hidden_feats//gat_n_heads, gat_n_heads)
                    if space_mp_type == 'gat' else
                    dglnn.SAGEConv(hidden_feats, hidden_feats, 'pool'),
                    # if space_mp_type == 'SAGE' else
                    # nn.Linear(hidden_feats, hidden_feats),
                'geom_bb': 
                    dglnn.GraphConv(hidden_feats, hidden_feats)
                    if space_mp_type == 'gcn' else
                    dglnn.GATConv(hidden_feats, hidden_feats//gat_n_heads, gat_n_heads)
                    if space_mp_type == 'gat' else
                    dglnn.SAGEConv(hidden_feats, hidden_feats, 'pool'),
                    # if space_mp_type == 'SAGE' else
                    # nn.Linear(hidden_feats, hidden_feats),
                },aggregate= hagg_type) 
             for _ in range(self.layer_num)]
            )
        
        self.out_lin = nn.Linear(hidden_feats, self.target_num)
        # self.out_lin_y = nn.Linear(hidden_feats, target_num)
        

        
    def forward(self, graph):

        b_h = F.leaky_relu(self.lin_block(graph.nodes['block'].data['feature']))
        n_h   = F.leaky_relu(self.lin_net(graph.nodes['net'].data['feature']))
        g_h   = F.leaky_relu(self.lin_grid(graph.nodes['grid'].data['feature']))

        graph.nodes['block'].data['h'] = F.leaky_relu(self.lin_block_2(b_h))
        graph.nodes['net'].data['h']   = F.leaky_relu(self.lin_net_2(n_h))
        graph.nodes['grid'].data['h']  = F.leaky_relu(self.lin_grid_2(g_h))


        for i in range(self.layer_num):
            h = self.gnn_layers[i](graph, graph.ndata['h'])
            if self.hagg_type == 'stack':
                h['block'] = h['block'].view(graph.num_nodes('block'),-1)
                h['net'] = h['net'].view(graph.num_nodes('net'),-1)
                h['grid'] = self.lin_grid_agg(h['grid'].view(graph.num_nodes('grid'),-1))
                h['grid'] = self.lin_grid_agg2(h['grid'])
            else:
                if self.netlist_mp_type == 'gat':
                    h['block'] = dgl.apply_each(h['block'], lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
                    h['net'] = dgl.apply_each(h['net'], lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
                if self.layout_mp_type =='gat':
                    h['grid'] = dgl.apply_each(h['grid'], lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))

            h = {k: F.relu(v) for k, v in h.items()}

        grid_embed = h['grid']
        pred_out = self.out_lin(h['grid'])
        return  pred_out, grid_embed 


if __name__ == '__main__':

    # benchmarks_list = ['neuron']

    # data_loader = DataLoader(circuit_list=benchmarks_list, seed_range=10)
    # train_dataset, test_dataset = data_loader.load()
    from dgl import load_graphs

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    graph_path = '/home/mavo/AA_RL_Placer/hetero_gnn_routability_titan/dataset/neuron/neuron_seed0_dgl_graph.bin'
    g, g_dict = load_graphs(graph_path)
    print(g[0].nodes['grid'].data['label'].shape)


    in_block_feats = g[0].nodes['block'].data['feature'].shape[1]
    in_net_feats = g[0].nodes['net'].data['feature'].shape[1]
    in_grid_feats = g[0].nodes['grid'].data['feature'].shape[1]

    model = FHGNN(in_block_feats, in_net_feats, in_grid_feats).to(DEVICE)

    pred_out, grid_embed  = model.forward(g[0].to(DEVICE))
    print(pred_out.shape)
    print(grid_embed.shape)


    pass
