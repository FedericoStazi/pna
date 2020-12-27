import torch.nn as nn
import torch
import dgl
from nets.gru import GRU
from nets.eig_layer import EIGLayer
from nets.mlp_readout_layer import MLPReadout




class EIGNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_feat = net_params['num_feat']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.JK = net_params['JK']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']
        self.device = device

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = nn.Linear(num_feat, hidden_dim)

        self.layers = nn.ModuleList([EIGLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                      edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
             in range(n_layers - 1)])
        self.layers.append(EIGLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat, edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)


        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem


    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        #h = self.in_feat_dropout(h)
        if self.JK == 'sum':
            h_list = [h]

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t
            if self.JK == 'sum':
                h_list.append(h)

        g.ndata['h'] = h

        if self.JK == 'last':
            g.ndata['h'] = h

        elif self.JK == 'sum':
            h = 0
            for layer in h_list:
                h += layer
            g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "directional_abs":
            g.ndata['dir'] = h * torch.abs(g.ndata['eig'][:, 1:2].to(self.device)) / torch.sum(
                torch.abs(g.ndata['eig'][:, 1:2].to(self.device)), dim=1, keepdim=True)
            hg = torch.cat([dgl.mean_nodes(g, 'dir'), dgl.mean_nodes(g, 'h')], dim=1)
        elif self.readout == "directional":
            g.ndata['dir'] = h * g.ndata['eig'][:, 1:2].to(self.device) / torch.sum(
                torch.abs(g.ndata['eig'][:, 1:2].to(self.device)), dim=1, keepdim=True)
            hg = torch.cat([torch.abs(dgl.mean_nodes(g, 'dir')), dgl.mean_nodes(g, 'h')], dim=1)
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        distances = []
        for s1 in scores:
            for s2 in scores:
                distances.append(abs(s1 - s2))
        print(torch.cuda.FloatTensor(distances).size())
        print(torch.cuda.FloatTensor(distances))
        print(targets.size())
        print(targets)
        loss = nn.MSELoss()(torch.cuda.FloatTensor(distances), targets)
        return loss