import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GCNNet(nn.Module):

    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, num_head, pred_dim, class_cardinality, dropout_rate, attention_dropout_rate, indices_neighbor):
        super(GCNNet, self).__init__()

        self.bn = nn.BatchNorm1d(out_dim)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.num_head = num_head
 
        self.activation = nn.ELU()

        self.block = GCNBlock(n_layer, in_dim, hidden_dim, out_dim, self.activation, num_head, dropout_rate, attention_dropout_rate, indices_neighbor)
    
        self.pred1 = Classifier(out_dim,
                                pred_dim,
                                self.activation,
                                dropout_rate)
        
        self.pred2 = Classifier(pred_dim,
                                class_cardinality,
                                None,
                                0)
        
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, adj, target_X, target, is_val, epoch):
        x = x.unsqueeze(0) 
        tmp_list = list()

        for i in range(self.num_head):
            tmp_list.append(x+0) 

        x = torch.cat(tmp_list, dim=0) 

        out, adj = self.block(x, adj, is_val, epoch)

        # Average multi-head
        out = torch.sum(out, dim=0) 
        out = torch.div(out, self.num_head) 

        out = self.bn(out)

        out = self.activation(out)

        if self.dropout_rate > 0:
            out = self.dropout(out)

        #classifier
        y = self.pred1(out)
        y = self.pred2(y)
       
        loss = self.loss(y[target_X], target)
    
        return loss, y[target_X]
 
class GCNBlock(nn.Module):

    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, act, num_head, dropout_rate, attention_dropout_rate, indices_neighbor):  
        super(GCNBlock, self).__init__()

        self.num_head = num_head 

        self.activation = act

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i == 0 else hidden_dim//(2*i),
                                        out_dim if i == n_layer-1 else hidden_dim//(2*(i+1)), 
                                        self.activation if i != n_layer-1 else None, 
                                        num_head,
                                        dropout_rate, attention_dropout_rate, indices_neighbor))
        
    def forward(self, x, adj, is_val, epoch):
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i == 0 else out), adj, is_val, epoch, i)

        return out, adj



class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, act, num_head, dropout_rate, attention_dropout_rate, indices_neighbor): 
        super(GCNLayer, self).__init__()

        self.num_head = num_head

        self.attention = Attention(out_dim, num_head, indices_neighbor, dropout_rate, attention_dropout_rate)
        self.activation = act

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.filter_matrix_list = nn.ModuleList()
        for i in range(self.num_head):
            self.filter_matrix_list.append(nn.Linear(in_dim, out_dim))
    
        for layer in self.filter_matrix_list:
            nn.init.xavier_uniform_(layer.weight)
    
        self.bn_list = nn.ModuleList()
        for i in range(self.num_head):
            self.bn_list.append(nn.BatchNorm1d(out_dim))
    
    def forward(self, x, adj, is_val, epoch, layer_position):
        num_head_chunks_of_x = torch.chunk(x, self.num_head, dim=0)

        x_list = list()
        
        for chunk in num_head_chunks_of_x:
            x_list.append(chunk.squeeze(0))

        tmp_out = list()

        for (bn, layer, chunk) in zip(self.bn_list, self.filter_matrix_list, x_list):
            tmp_out.append(bn(layer(chunk)))

        tmp_out2 = list()
        for chunk in tmp_out:
            tmp_out2.append(chunk.unsqueeze(0))

        out = torch.cat(tmp_out2, dim=0) 

        out = self.attention(out, adj, is_val, epoch, layer_position)

        if self.activation != None:
            out = self.activation(out) 

        if self.dropout_rate > 0:
            out = self.dropout(out)

        return out, adj 

class Attention(nn.Module):

    def __init__(self, out_dim, num_head, indices_neighbor, dropout_rate, attention_dropout_rate):
        super(Attention, self).__init__()

        self.indices_neighbor = indices_neighbor

        self.num_head = num_head
        self.out_dim = out_dim

        self.att_layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.softmax = nn.Softmax(dim=-1)

        self.attention_dropout_rate = attention_dropout_rate

        for i in range(self.num_head):
            self.att_layers.append(nn.Linear(self.out_dim, len(self.indices_neighbor))) 

        for layer in self.att_layers:
            nn.init.xavier_uniform_(layer.weight)
           
    def forward(self, x, adj, is_val, epoch, layer_position):
        shape = list(x.size()) 

        heads = list() 

        x_chunks = torch.chunk(x, self.num_head, dim=0) 

        for i in range(self.num_head):
            x_transformed = x_chunks[i].squeeze(0) 

            no_concatenated_wh = self.att_layers[i](x_transformed)

            att_score = self.leaky_relu(no_concatenated_wh)

            results_of_softmax = []
            for i, index_neighbor in enumerate(self.indices_neighbor):
                target_row = att_score[i].detach()
                candidate_of_softmax = target_row[index_neighbor] 

                results_of_softmax.append(self.softmax(candidate_of_softmax)) 

            att_ratio = []
            for index_neighbor, result_of_softmax in zip(self.indices_neighbor, results_of_softmax):
                result_row = np.zeros((1, shape[1])).squeeze(0) 
    
                result_row[index_neighbor] = result_of_softmax.cpu().numpy()

                att_ratio.append(torch.from_numpy(result_row).unsqueeze(0))

            att_ratio = torch.cat(att_ratio, dim=0).type(torch.cuda.FloatTensor) 

            att_ratio = F.dropout(att_ratio, self.attention_dropout_rate, training=self.training)

            x_head = torch.matmul(att_ratio, x_transformed) 

            heads.append(x_head.unsqueeze(0)) 

        out = torch.cat(heads, dim=0) 

        return out

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, act, dropout_rate):
        super(Classifier, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.bn = nn.BatchNorm1d(out_dim)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.activation = act

    def forward(self, x):
        out = self.linear(x)

        if self.dropout_rate > 0:
            out = self.bn(out)
     
        if self.activation != None:
            out = self.activation(out)

        if self.dropout_rate > 0:
            out = self.dropout(out)
    
        return out






