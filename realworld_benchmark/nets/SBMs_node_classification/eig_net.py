import torch.nn as nn
import torch
import dgl
from nets.gru import GRU
from nets.eig_layer import EIGLayer
from nets.mlp_readout_layer import MLPReadout




class EIGNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.pos_enc_dim = net_params['pos_enc_dim']
        if self.pos_enc_dim > 0:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.n_classes = n_classes
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.JK = net_params['JK']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']
        
        #self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([EIGLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                      edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
             in range(n_layers - 1)])
        self.layers.append(EIGLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                                    edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)
        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)
            
        self.MLP_layer = MLPReadout(out_dim, n_classes)
       

            

    def forward(self, g, h, e, snorm_n, snorm_e):
        print(h)
        #h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc_dim > 0:
            h_pos_enc = self.embedding_pos_enc(g.ndata['pos_enc'].to(self.device))
            h = h + h_pos_enc
        if self.JK == 'sum':
            h_list = [h]

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t
            if self.JK == 'sum':
                h_list.append(h)

        if self.JK == 'last':
            pass

        elif self.JK == 'sum':
            h = 0
            for layer in h_list:
                h += layer

        h_out = self.MLP_layer(h)

        return h_out

    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss



