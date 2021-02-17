import os
import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from copy import deepcopy
from itertools import chain

from transformers import BertTokenizer, RobertaTokenizer
from tqdm import tqdm

MAX_NODE_LENGTH=10
# for xWant ,neg_prop=4, all_in_aser, when this equals 10, filter out 0.6%. when 11, filter out 0.1%.


np.random.seed(229)

class InferenceGraphDataset():
    def __init__(self, np_file_path, device, encoder, training_graph):

        self.training_graph = training_graph

        self.data = np.load(np_file_path, allow_pickle=True)[()]

        # train_edges = dict([(tuple(edge), True) for edge in self.training_graph.train_edges])

        # self.node2id = dict([(node, i) for i, node in enumerate(self.data.flatten())])
        # self.id2node = dict([(i, node) for i, node in enumerate(self.data.flatten())])

        self.data_id = {}

        self.data_id["head"] = np.array([[self.training_graph.node2id[head], 
            self.training_graph.node2id[tail], hid] for head, tail, hid in self.data["head"] \
            if len(head.split()) < MAX_NODE_LENGTH and len(tail.split()) < MAX_NODE_LENGTH]) # if (self.training_graph.node2id[head], self.training_graph.node2id[tail]) not in train_edges

        self.data_id["tail"] = np.array([[self.training_graph.node2id[head], 
            self.training_graph.node2id[tail], -1] for head, tail in self.data["tail"] \
            if len(head.split()) < MAX_NODE_LENGTH and len(tail.split()) < MAX_NODE_LENGTH]) # if (self.training_graph.node2id[head], self.training_graph.node2id[tail]) not in train_edges

        self.data_id["new"] = np.array([[self.training_graph.node2id[head], 
            self.training_graph.node2id[tail], -1] for head, tail in self.data["new"] \
            if len(head.split()) < MAX_NODE_LENGTH and len(tail.split()) < MAX_NODE_LENGTH]) # if (self.training_graph.node2id[head], self.training_graph.node2id[tail]) not in train_edges

    def get_nodes_tokenized(self):
        return self.training_graph.get_nodes_tokenized()
    def get_adj_list(self):
        return self.training_graph.get_adj_list()
    def get_batch(self, batch_size=16, mode="head"):
        for i in range(0, len(self.data_id[mode]), batch_size):
            yield self.data_id[mode][i:min(i+batch_size, len(self.data_id[mode]))]

class InferenceSimpleDataset():
    def __init__(self, np_file_path, device, encoder,training_graph ):
        self.training_graph = training_graph
        self.data = np.load(np_file_path, allow_pickle=True)[()]
        # train_edges = dict([(tuple(edge), True) for edge in self.training_graph.train_edges])

        self.data_id = {}

        self.data_id["head"] = np.array([[self.training_graph.node2id[head], 
            self.training_graph.node2id[tail], hid] for head, tail, hid in self.data["head"] \
            if len(head.split()) < MAX_NODE_LENGTH and len(tail.split()) < MAX_NODE_LENGTH ]) 
        self.data_id["tail"] = np.array([[self.training_graph.node2id[head], 
            self.training_graph.node2id[tail], -1] for head, tail in self.data["tail"] \
            if len(head.split()) < MAX_NODE_LENGTH and len(tail.split()) < MAX_NODE_LENGTH ]) 
        self.data_id["new"] = np.array([[self.training_graph.node2id[head], 
            self.training_graph.node2id[tail], -1] for head, tail in self.data["new"] \
            if len(head.split()) < MAX_NODE_LENGTH and len(tail.split()) < MAX_NODE_LENGTH ]) 

            # if (self.training_graph.node2id[head], self.training_graph.node2id[tail]) not in train_edges
    def get_nodes_tokenized(self):
        return self.training_graph.get_nodes_tokenized()
    def get_batch(self, batch_size=16, mode="head"):
        for i in range(0, len(self.data_id[mode]), batch_size):
            yield self.data_id[mode][i:min(i+batch_size, len(self.data_id[mode]))]

class GraphDataset():

    def __init__(self, nx_file_path, 
        device,
        encoder,
        split=[0.8, 0.1, 0.1],
        max_train_num=1000000,
        load_edge_types="ATOMIC",
        negative_sample="fix_head",
        atomic_csv_path="/home/tfangaa/Downloads/ATOMIC/v4_atomic_all_agg.csv",
        random_split=False,
        neg_prop=1.0):
        assert load_edge_types in ["ATOMIC", "ASER", "ATOMIC+ASER"], \
            "should be in [\"ATOMIC\", \"ASER\", \"ATOMIC+ASER\"]"

        """
            load_edge_types controls the edges to be loaded to self.adj_list
        """

        # 1. Load graph
        self.atomic_csv_path = atomic_csv_path
        G = nx.read_gpickle(nx_file_path)

        print("dataset statistics:\nnumber of nodes:{}\nnumber of edges:{}\n".format(len(G.nodes()), len(G.edges())))
        
        self.id2node = {}
        self.node2id = {}

        filter_nodes = []
        for node in G.nodes():
            if len(node.split()) > MAX_NODE_LENGTH: # filter extra large nodes
                filter_nodes.append(node)
        print("num of removing nodes:", len(filter_nodes))                
        G.remove_nodes_from(filter_nodes)


        for i, node in enumerate(G.nodes()):
            self.id2node[i] = node
            self.node2id[node] = i

        # 2. Prepare training and testing edges
        all_edges_shuffle = list(G.edges.data())
        np.random.shuffle(all_edges_shuffle)
        ATOMIC_edges = [edge for edge in all_edges_shuffle if edge[2]["relation"]=="ATOMIC"]
        atomic_edge_cnter = Counter([(edge[2]['hid'], edge[2]['tid']) for edge in ATOMIC_edges])
        self.train_id, self.val_id, self.test_id = [int(s*len(ATOMIC_edges)) for s in np.cumsum(split)/np.sum(split)]

        edge_by_htid = dict()
        for e in ATOMIC_edges:
            htid = (e[2]['hid'], e[2]['tid'])
            if htid not in edge_by_htid:
                edge_by_htid[htid] = [[self.node2id[e[0]], self.node2id[e[1]]]]
            else:
                edge_by_htid[htid].append([self.node2id[e[0]], self.node2id[e[1]]])

        # randomly select edges
        current_edge_list = deepcopy(list(atomic_edge_cnter.keys()))
        if random_split:
            train_edges_pos_all, val_edges_pos_all, test_edges_pos_all = self.get_random_split(current_edge_list, edge_by_htid)
        else:
            train_edges_pos_all, val_edges_pos_all, test_edges_pos_all = self.get_split_from_atomic(edge_by_htid)

        print('Number of positive training examples:{}, validating:{}, testing:{}'.format(len(train_edges_pos_all), len(val_edges_pos_all), len(test_edges_pos_all)))

        if len(train_edges_pos_all) > max_train_num:
            train_edges_pos = list(np.array(train_edges_pos_all)[np.random.permutation(len(train_edges_pos_all))[:max_train_num]])
            val_edges_pos = list(np.array(val_edges_pos_all)[np.random.permutation(len(val_edges_pos_all))[:int(max_train_num * split[1]/split[0])]])
            test_edges_pos = list(np.array(test_edges_pos_all)[np.random.permutation(len(test_edges_pos_all))[:int(max_train_num * split[2]/split[0])]])
        else:
            train_edges_pos = train_edges_pos_all
            val_edges_pos = val_edges_pos_all
            test_edges_pos = test_edges_pos_all

        edge_dict = dict([((self.node2id[head], self.node2id[tail]), True) for head, tail in G.edges()])

        print('Number of positive training examples after trucating:{}, validating:{}, testing:{}'.format(len(train_edges_pos), len(val_edges_pos), len(test_edges_pos)))

        # 3. Sample negative edges

        if negative_sample == "fix_head":
            # bipartite graph
            all_heads = [self.node2id[node] for node, out_degree in G.out_degree if out_degree>0]
            all_tails = [self.node2id[node] for node, out_degree in G.out_degree if out_degree==0]
            neg_edges = []
            num_neg = len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos)
            for i in range( int(num_neg * neg_prop) ):
                hd_idx = np.random.randint(0, len(all_heads))
                tl_idx = np.random.randint(0, len(all_tails))
                while (all_heads[hd_idx], all_tails[tl_idx]) in edge_dict:
                    hd_idx = np.random.randint(0, len(all_heads))
                    tl_idx = np.random.randint(0, len(all_tails))
                neg_edges.append([all_heads[hd_idx], all_tails[tl_idx]])
        elif negative_sample == "from_all":
            neg_edges = []
            num_neg = len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos)
            for i in range( int(num_neg * neg_prop) ):
                rnd = np.random.randint(0, len(self.node2id), 2)
                tmp_edge = (rnd[0], rnd[1])
                while tmp_edge in edge_dict or tmp_edge[0] == tmp_edge[1]:
                    rnd = np.random.randint(0, len(self.node2id), 2)
                    tmp_edge = (rnd[0], rnd[1])
                neg_edges.append(list(tmp_edge))
        elif negative_sample == "prepared_neg":
            # some of the negative samples are pre-prepared
            neg_train = [[self.node2id[head], self.node2id[tail]] \
                for head, tail, feat in G.edges.data() if feat["relation"]=="neg_trn"]
            neg_val = [[self.node2id[head], self.node2id[tail]] \
                for head, tail, feat in G.edges.data() if feat["relation"]=="neg_dev"]
            neg_test = [[self.node2id[head], self.node2id[tail]] \
                for head, tail, feat in G.edges.data() if feat["relation"]=="neg_tst"]
            print("num of prepared neg for train:{}, dev:{}, test:{}".format(len(neg_train), len(neg_val), len(neg_test)))
            neg_edges = []
            num_neg = len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos)
            for i in range(int(num_neg * neg_prop) - len(neg_train) - len(neg_val) - len(neg_test) ):
                rnd = np.random.randint(0, len(self.node2id), 2)
                tmp_edge = (rnd[0], rnd[1])
                while tmp_edge in edge_dict or tmp_edge[0] == tmp_edge[1]:
                    rnd = np.random.randint(0, len(self.node2id), 2)
                    tmp_edge = (rnd[0], rnd[1])
                neg_edges.append(list(tmp_edge))
            trn_val_idx = int(len(train_edges_pos)*neg_prop) - len(neg_train)
            val_tst_idx = trn_val_idx + int(len(val_edges_pos)*neg_prop)-len(neg_val)
            neg_edges = neg_train + neg_edges[:trn_val_idx]\
                       +neg_val + neg_edges[trn_val_idx:val_tst_idx]\
                       +neg_test + neg_edges[val_tst_idx:]
            assert len(neg_edges) == int((len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos))*neg_prop)

        train_edges_neg = neg_edges[:int(len(train_edges_pos)*neg_prop)]
        val_edges_neg = neg_edges[int(len(train_edges_pos)*neg_prop):int((len(train_edges_pos)+len(val_edges_pos))*neg_prop)]
        test_edges_neg = neg_edges[int((len(train_edges_pos)+len(val_edges_pos))*neg_prop):]
        print('Number of negative examples after trucating:{}, validating:{}, testing:{}'.format(len(train_edges_neg), len(val_edges_neg), len(test_edges_neg)))

        self.train_labels = np.array([0] * len(train_edges_neg) + [1] * len(train_edges_pos))
        self.train_edges = np.array(train_edges_neg + train_edges_pos)
        train_shuffle_idx = np.random.permutation(len(self.train_edges))
        self.train_labels, self.train_edges = self.train_labels[train_shuffle_idx], self.train_edges[train_shuffle_idx]

        self.val_labels = np.array([0] * len(val_edges_neg) + [1] * len(val_edges_pos))
        self.val_edges = np.array(val_edges_neg + val_edges_pos)
        val_shuffle_idx = np.random.permutation(len(self.val_edges))
        self.val_labels, self.val_edges = self.val_labels[val_shuffle_idx], self.val_edges[val_shuffle_idx]

        self.test_labels = np.array([0] * len(test_edges_neg) + [1] * len(test_edges_pos))
        self.test_edges = np.array(test_edges_neg + test_edges_pos)
        test_shuffle_idx = np.random.permutation(len(self.test_edges))
        self.test_labels, self.test_edges = self.test_labels[test_shuffle_idx], self.test_edges[test_shuffle_idx]

        print('finish preparing neg samples')

        self.mode_edges = {
            "train":torch.tensor(self.train_edges).to(device),
            "valid":torch.tensor(self.val_edges).to(device),
            "test":torch.tensor(self.test_edges).to(device)
        }
        self.mode_labels = {
            "train":torch.tensor(self.train_labels).to(device),
            "valid":torch.tensor(self.val_labels).to(device),
            "test":torch.tensor(self.test_labels).to(device)
        }

        # Prepare a sparse adj matrix, mask all the valid and test set
        # adj list that contains all the training edges
        self.adj_list = [[] for i in range(len(self.id2node))]

        # Edges are all the edges except for those in test/val set
        val_edges_dict = dict([((edge[0], edge[1]), True) for edge in val_edges_pos])
        test_edges_dict = dict([((edge[0], edge[1]), True) for edge in test_edges_pos])

        for head, tail, feat in G.edges.data():
            if load_edge_types == "ATOMIC":
                if feat["relation"] != "ATOMIC":
                    continue
            elif load_edge_types == "ASER":
                if feat["relation"] != "ASER":
                    continue
            elif load_edge_types == "ATOMIC+ASER":
                pass
            if (self.node2id[head], self.node2id[tail]) not in val_edges_dict \
                and (self.node2id[head], self.node2id[tail]) not in test_edges_dict :
                self.adj_list[self.node2id[head]].append(self.node2id[tail])

        # 4. Tokenize nodes

        if encoder == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif encoder == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.id2nodestoken = dict([(self.node2id[line], torch.tensor(self.tokenizer.encode(line, 
            add_special_tokens=True)).to(device)) for line in tqdm(self.node2id)])

    def get_random_split(self, current_edge_list, edge_by_htid):
        train_edges_pos_all = []
        covered_atomic_edge = {}

        while len(train_edges_pos_all) <= self.train_id:
            # select an edge
            rnd_id = np.random.randint(len(current_edge_list))
            htid = current_edge_list[rnd_id]
            current_edge_list.pop(rnd_id)
            train_edges_pos_all.extend(edge_by_htid[htid])

        val_edges_pos_all = []
        while len(train_edges_pos_all) + len(val_edges_pos_all) <= self.val_id:
            rnd_id = np.random.randint(len(current_edge_list))
            htid = current_edge_list[rnd_id]
            current_edge_list.pop(rnd_id)
            val_edges_pos_all.extend(edge_by_htid[htid])

        test_edges_pos_all = []
        while len(train_edges_pos_all) + len(val_edges_pos_all) + len(test_edges_pos_all) < self.test_id:
            rnd_id = np.random.randint(len(current_edge_list))
            htid = current_edge_list[rnd_id]
            current_edge_list.pop(rnd_id)
            test_edges_pos_all.extend(edge_by_htid[htid])
        return train_edges_pos_all, val_edges_pos_all, test_edges_pos_all
    def get_split_from_atomic(self, edge_by_htid):
        train_edges_pos_all = []
        val_edges_pos_all = []
        test_edges_pos_all = []
        atomic_raw = pd.read_csv(self.atomic_csv_path)
        splits = dict((i,spl) for i,spl in enumerate(atomic_raw['split']))
        for htid, edge in edge_by_htid.items():
            if splits[htid[0]] == "trn":
                train_edges_pos_all.extend(edge)
            elif splits[htid[0]] == "dev":
                val_edges_pos_all.extend(edge)
            elif splits[htid[0]] == "tst":
                test_edges_pos_all.extend(edge)
        return train_edges_pos_all, val_edges_pos_all, test_edges_pos_all

    def get_adj_list(self):
        return self.adj_list
    def get_nid2text(self):
        return self.id2node    

    def get_nodes_tokenized(self):
        return self.id2nodestoken

    def get_batch(self, batch_size=16, mode="train"):
        assert mode in ["train", "valid", "test"], "invalid mode"
        
        for i in range(0, len(self.mode_edges[mode]), batch_size):
            yield self.mode_edges[mode][i:min(i+batch_size, len(self.mode_edges[mode]))], \
                self.mode_labels[mode][i:min(i+batch_size, len(self.mode_edges[mode]))]