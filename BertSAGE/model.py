import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, RobertaModel
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
import numpy as np

MAX_SEQ_LENGTH=30

def eval(data_loader, model, test_batch_size, criterion, mode="test", metric="acc"):
    loss = 0
    correct_num = 0
    total_num = 0
    model.eval()
    num_steps = 0
    # the accuracy of positive examples
    correct_pos = 0
    total_pos = 0

    with torch.no_grad():
        for batch in data_loader.get_batch(batch_size=test_batch_size, mode=mode):
            edges, labels = batch
            b_s, _ = edges.shape # batch_size, 2
            all_nodes = edges.reshape([-1])

            logits = model(all_nodes, b_s) # (batch_size, 2)

            loss += criterion(logits, labels).item()

            predicted = torch.max(logits, dim=1)[1]
            correct_num += (predicted == labels).sum().item()
            total_num += b_s
            num_steps += 1

            correct_pos += ( (predicted == labels) & (labels == 1)).sum().item()
            total_pos += (labels == 1).sum().item()


            # print("eval", labels, logits)
    model.train()
    # print(mode+" set accuracy:", correct_num / total_num, "loss:", loss/num_steps)
    # return F1,
    TP = correct_pos
    FN = total_pos - correct_pos
    R = TP / (TP+FN)
    FP = total_num - correct_num - FN
    P = TP / (TP+FP)
    # return 2*P*R/(P+R), correct_pos/total_pos

    if metric == "acc":
        return correct_num / total_num, correct_pos/total_pos
    elif metric == "f1":
        return 2*P*R/(P+R), correct_pos/total_pos

class LinkPrediction(nn.Module):
    def __init__(self, encoder, adj_lists, nodes_tokenized, device, num_layers=1,num_neighbor_samples=10):
        super(LinkPrediction, self).__init__()

        self.graph_model = GraphSage(
                        encoder=encoder,
                        num_layers=num_layers, 
                      input_size=768, 
                      output_size=768, 
                      adj_lists=adj_lists,
                      nodes_tokenized=nodes_tokenized,
                      device=device,
                      agg_func='MEAN',
                      num_neighbor_samples=num_neighbor_samples)

        self.link_classifier = Classification(768*2, 2, device)

    def forward(self, all_nodes, b_s):
        embs = self.graph_model(all_nodes)# (2*batch_size, emb_size)
        logits = self.link_classifier(embs.view([b_s, -1])) # (batch_size, 2*emb_size)

        return logits

class SimpleClassifier(nn.Module):
    def __init__(self, encoder, adj_lists, nodes_tokenized, device):
        super(SimpleClassifier, self).__init__()
        self.nodes_tokenized = nodes_tokenized
        self.device = device
        if encoder == "bert":
            self.roberta_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        elif encoder == "roberta":
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)

        self.link_classifier = Classification(768*2, 2, device)

    def get_roberta_embs(self, input_ids):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.roberta_model(input_ids)
        return torch.mean(outputs[0], dim=1) # aggregate embs

    def forward(self, all_nodes, b_s):
        embs = self.get_roberta_embs(
            pad_sequence([self.nodes_tokenized[int(node)] for node in all_nodes], padding_value=1).transpose(0, 1).to(self.device)
        )

        logits = self.link_classifier(embs.view([b_s, -1])) # (batch_size, 2*emb_size)

        return logits


class Classification(nn.Module):

    def __init__(self, emb_size, num_classes, device):
        super(Classification, self).__init__()

        #self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
        self.linear = nn.Linear(emb_size, num_classes).to(device)

    def forward(self, embs):
        logists = self.linear(embs)
        return logists        

class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_size, out_size): 
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.linear = nn.Linear(self.input_size*2, self.out_size)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes    -- list of nodes
        """
        combined = torch.cat([self_feats, aggregate_feats], dim=1)
        # [b_s, emb_size * 2]
        combined = F.relu( self.linear(combined) ) # [b_s, emb_size]
        return combined

class GraphSage(nn.Module):
    """docstring for GraphSage"""
    def __init__(self, encoder, num_layers, input_size, output_size, 
        adj_lists, nodes_tokenized, device, agg_func='MEAN', num_neighbor_samples=10):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.agg_func = agg_func
        self.num_neighbor_samples = num_neighbor_samples

        if encoder == "bert":
            self.roberta_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        elif encoder == "roberta":
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)

        self.adj_lists = adj_lists
        self.nodes_tokenized = nodes_tokenized

        for index in range(1, num_layers+1):
            layer_size = self.out_size if index != 1 else self.input_size
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, self.out_size).to(device))
        # self.fill_tensor = torch.FloatTensor(1, 768).fill_(0).to(self.device)
        self.fill_tensor = torch.nn.Parameter(torch.rand(1, 768)).to(self.device)



    def get_roberta_embs(self, input_ids):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.roberta_model(input_ids)
        return torch.mean(outputs[0], dim=1) # aggregate embs


    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch -- (list: ids)batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch) # node idx

        nodes_batch_layers = [(lower_layer_nodes,)]
        
        for i in range(self.num_layers):
            lower_layer_neighs, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes,  num_sample=self.num_neighbor_samples)
            # lower_layer_neighs: list(list())
            # lower_layer_nodes: list(nodes of next layer)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_layer_neighs))

        all_nodes = np.unique([int(n) for n in list(chain(*[layer[0] for layer in nodes_batch_layers]))])
        all_nodes_idx = dict([(node, idx) for idx, node in enumerate(all_nodes) ])


        all_neigh_nodes = pad_sequence([self.nodes_tokenized[node ] for node in all_nodes], padding_value=1).transpose(0, 1)[:, :MAX_SEQ_LENGTH].to(self.device)

        pre_hidden_embs = self.get_roberta_embs(
            all_neigh_nodes
        )

        # (num_all_node, emb_size)

        for layer_idx in range(1, self.num_layers+1):
            this_layer_nodes = nodes_batch_layers[layer_idx][0] # all nodes in this layer
            neigh_nodes, neighbors_list = nodes_batch_layers[layer_idx-1] # previous layer
            # list(), list(list())

            aggregate_feats = self.aggregate(neighbors_list, pre_hidden_embs, all_nodes_idx)
            # (this_layer_nodes_num, emb_size)

            sage_layer = getattr(self, 'sage_layer'+str(layer_idx))
            
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]], #pre_hidden_embs[layer_nodes],
                                        aggregate_feats=aggregate_feats)

            # cur_hidden_embs = torch.cat([pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]].unsqueeze(1), 
                                    # aggregate_feats.unsqueeze(1)], dim=1) # (b_s, 2, emb_size)
            # cur_hidden_embs = torch.mean(cur_hidden_embs, dim=1)

            pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]] = cur_hidden_embs

        # (input_batch_node_size, emb_size)
        # output the embeddings of the input nodes
        return pre_hidden_embs[[all_nodes_idx[int(n)] for n in nodes_batch]]

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        # TODO
        neighbors_list = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            samp_neighs = [np.random.choice(neighbors, num_sample) if len(neighbors)>0 else [] for neighbors in neighbors_list]
        else:
            samp_neighs = neighbors_list
        _unique_nodes_list = np.unique(list(chain(*samp_neighs)))
        return samp_neighs, _unique_nodes_list

    def aggregate(self, neighbors_list, pre_hidden_embs, all_nodes_idx):
        if self.agg_func == 'MEAN':
            agg_list = [torch.mean(pre_hidden_embs[ [int(all_nodes_idx[n]) for n in neighbors] ], dim=0).unsqueeze(0)\
              if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, pre_hidden_embs.shape[1]).fill_(0).to(self.device)
        if self.agg_func == 'MAX':
            return 0
