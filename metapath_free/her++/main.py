import torch
import torch.nn as nn

import pickle

import argparse

import numpy as np

from utils import f1_score, accuracy

from model import GCNNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=30,
                        help='Training Epochs')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='out dimension of Task model')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay factor for l2 regularization')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of task model layer')
    parser.add_argument('--num_head', type=int, default=3,
                        help='number of attention head')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='hidden dimension')
    parser.add_argument('--pred_dim', type=int, default=16,
                        help='prediction dimension for classifier')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5,
                        help='attention dropout rate')
    parser.add_argument('--count', type=int, default=5,
                        help='count -> number of experiments')

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    out_dim = args.out_dim
    num_head = args.num_head
    lr = args.lr
    weight_decay = args.weight_decay
    n_layer = args.n_layer
    pred_dim = args.pred_dim
    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate
    attention_dropout_rate = args.attention_dropout_rate
    count = args.count
    dataset = args.dataset

    with open('data/'+dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
    
    num_nodes = edges[0].shape[0]

    A = list()

    for edge in edges:
        A.append(torch.from_numpy(edge.todense()).type(torch.FloatTensor))
    A.append(torch.eye(num_nodes).type(torch.FloatTensor))

    pa_indices_neighbor = []
    ap_indices_neighbor = []
    ps_indices_neighbor = []
    sp_indices_neighbor = []
    i_indices_neighbor = []
    
    for i in range(num_nodes):
        pa_indices_neighbor.append([]*1)
        ap_indices_neighbor.append([]*1)
        ps_indices_neighbor.append([]*1)
        sp_indices_neighbor.append([]*1)
        i_indices_neighbor.append([]*1)
    
    for node_connect_info in A[0].nonzero():
        pa_indices_neighbor[node_connect_info[0]].append(int(node_connect_info[1]))

    for node_connect_info in A[1].nonzero():
        ap_indices_neighbor[node_connect_info[0]].append(int(node_connect_info[1]))

    for node_connect_info in A[2].nonzero():
        ps_indices_neighbor[node_connect_info[0]].append(int(node_connect_info[1]))

    for node_connect_info in A[3].nonzero():
        sp_indices_neighbor[node_connect_info[0]].append(int(node_connect_info[1]))

    for node_connect_info in A[4].nonzero():
        i_indices_neighbor[node_connect_info[0]].append(int(node_connect_info[1]))

    indices_neighbor = []

    indices_neighbor.append(pa_indices_neighbor)
    indices_neighbor.append(ap_indices_neighbor)
    indices_neighbor.append(ps_indices_neighbor)
    indices_neighbor.append(sp_indices_neighbor)
    indices_neighbor.append(i_indices_neighbor)

    num_edge = len(indices_neighbor)

    node_features = torch.from_numpy(node_features)    
    node_features = node_features.type(torch.FloatTensor)

    in_dim = node_features.shape[1]

    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)

    class_cardinality = torch.max(train_target).item() + 1 

    for cnt in range(count):
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        best_epoch = 0

        model = GCNNet(n_layer,
                    in_dim,
                    hidden_dim,
                    out_dim,
                    num_head,
                    pred_dim,
                    class_cardinality,
                    dropout_rate,
                    attention_dropout_rate,
                    indices_neighbor,
                    num_edge)

        optimizer = torch.optim.Adam([
                                    {'params':model.parameters()}
                                ], lr=lr, weight_decay=weight_decay)

      
        for i in range(epochs):
            print('Epoch: ',i+1)

            model.train()
            model.zero_grad()

            loss, y_train = model(node_features, A, train_node, train_target)

            loss.backward()
            optimizer.step()

            train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=class_cardinality)).numpy()
            print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().numpy(), train_f1))

            model.eval()
            
            with torch.no_grad():
                val_loss, y_valid = model.forward(node_features, A, valid_node, valid_target)

                test_loss, y_test = model.forward(node_features, A, test_node, test_target)

                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=class_cardinality)).numpy()
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().numpy(), val_f1))

                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=class_cardinality)).numpy()
                test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().numpy(), test_f1, test_acc))

                if val_f1 > best_val_f1:
                    best_val_loss = val_loss.detach().numpy()
                    best_test_loss = test_loss.detach().numpy()
                    best_train_loss = loss.detach().numpy()
                    best_train_f1 = train_f1
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1
                    best_epoch = i

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
   
