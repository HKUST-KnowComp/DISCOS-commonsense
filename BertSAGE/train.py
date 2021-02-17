import warnings
warnings.filterwarnings("ignore")
import torch
import os
from dataloader import *
from model import *
import argparse
import pickle

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='simple', type=str, required=False,
                    choices=["graphsage", "simple"],
                    help="choose model")
parser.add_argument("--encoder", default='bert', type=str, required=False,
                    choices=["bert", "roberta"],
                    help="choose encoder")
parser.add_argument("--num_layers", default=1, type=int, required=False,
                    help="number of graphsage layers")
parser.add_argument("--lr", default=0.01, type=float, required=False,
                    help="learning rate")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 2000 steps")
parser.add_argument("--decay_every", default=500, type=int, required=False,
                    help="show test result every x steps")
parser.add_argument("--test_every", default=250, type=int, required=False,
                    help="show test result every x steps")
parser.add_argument("--batch_size", default=64, type=int, required=False,
                    help="batch size")
parser.add_argument("--epochs", default=3, type=int, required=False,
                    help="batch size")
parser.add_argument("--num_neighbor_samples", default=4, type=int, required=False,
                    help="num neighbor samples in GraphSAGE")
parser.add_argument("--load_edge_types", default='ATOMIC', type=str, required=False,
                    choices=["ATOMIC", "ASER", "ATOMIC+ASER"],
                    help="load what edges to data_loader.adj_lists")
parser.add_argument("--graph_cach_path", default="graph_cache/neg_{}_{}_{}_{}.pickle", 
                    type=str, required=False,
                    help="path of graph cache")
parser.add_argument("--optimizer", default='SGD', type=str, required=False,
                    choices=["SGD", "ADAM"],
                    help="optimizer to be used")
parser.add_argument("--negative_sample", default='from_all', type=str, required=False,
                    choices=["prepared_neg", "from_all", "fix_head"],
                    help="nagative sample methods")
parser.add_argument("--file_path", default='', type=str, required=True,
                    help="load training graph pickle")
parser.add_argument("--metric", default='acc', type=str, required=False,
                    choices=["f1", "acc"],
                    help="evaluation metric, either f1 or acc")
parser.add_argument("--neg_prop", default=1.0, type=float, required=False,
                    help="the proportion of negative sample: num_neg/num_pos")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
lr = args.lr
show_step = args.test_every
batch_size= args.batch_size
num_epochs = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_batch_size = 64
neg_prop = args.neg_prop

file_path = args.file_path

graph_cache = args.graph_cach_path.format(args.negative_sample, args.load_edge_types, os.path.basename(file_path).split(".")[0])
if not os.path.exists("models"):
    os.mkdir("models")
model_dir = "models/"+os.path.basename(file_path).split(".")[0]
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if args.model == "simple":
    model_save_path = os.path.join(model_dir, '{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_{}.pth'\
    .format(args.model, args.encoder, batch_size, args.optimizer, 
        args.lr, args.lrdecay, args.decay_every, args.metric))
elif args.model == "graphsage":
    model_save_path = os.path.join(model_dir, '{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_layer{}_neighnum_{}_graph_{}_{}.pth'\
                        .format(args.model, args.encoder, batch_size, args.optimizer, args.lr, 
                            args.lrdecay, args.decay_every, args.num_layers, 
                            args.num_neighbor_samples, args.load_edge_types, args.metric))

print(graph_cache)
if not os.path.exists(graph_cache):
    data_loader = GraphDataset(file_path, device, args.encoder, 
        negative_sample=args.negative_sample, load_edge_types=args.load_edge_types,
        neg_prop=neg_prop)
    with open(graph_cache, "wb") as writer:
        pickle.dump(data_loader,writer,pickle.HIGHEST_PROTOCOL)  
    print("after dumping graph cache to", graph_cache)
else:
    with open(graph_cache, "rb") as reader:
        data_loader = pickle.load(reader)
    print("after loading graph cache from", graph_cache)
    

if args.model == "simple":
    model = SimpleClassifier(encoder=args.encoder,
                        adj_lists=data_loader.get_adj_list(), 
                       nodes_tokenized=data_loader.get_nodes_tokenized(),
                       device=device,
                        )
elif args.model == 'graphsage':
    model = LinkPrediction(encoder=args.encoder,
                        adj_lists=data_loader.get_adj_list(), 
                       nodes_tokenized=data_loader.get_nodes_tokenized(),
                       device=device,
                       num_layers=args.num_layers,
                       num_neighbor_samples=args.num_neighbor_samples,
                        )

criterion = torch.nn.CrossEntropyLoss()
if args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif args.optimizer == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()

step = 0

best_valid_acc = 0
best_test_acc = 0   
best_valid_pos_acc = 0
best_test_pos_acc = 0 

my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lrdecay)

for epoch in range(num_epochs):
    for batch in data_loader.get_batch(batch_size=batch_size, mode="train"):
        # torch.cuda.empty_cache()
        step += 1
        if step % args.decay_every == 0:
            # lr = lr * args.lrdecay
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            my_lr_scheduler.step()
        # batch list((node_id1, node_id2))
        edges, labels = batch
        b_s, _ = edges.shape # batch_size, 2
        all_nodes = edges.reshape([-1])

        logits = model(all_nodes, b_s)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        # evaluate
        if step % show_step == 0:
            val_acc, val_pos_acc = eval(data_loader, model, test_batch_size, criterion, "valid", args.metric)
            test_acc, test_pos_acc = eval(data_loader, model, test_batch_size, criterion, "test", args.metric)
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
                best_test_acc = test_acc
                best_valid_pos_acc = val_pos_acc
                best_test_pos_acc = test_pos_acc
                
                torch.save(model.state_dict(), model_save_path)
                
            print(args.metric, ": epoch {}, step {}, current valid: {},"
                  "current test: {}, curret valid pos:{},"
                  " current test pos: {},".format(epoch, step, val_acc, test_acc, val_pos_acc, test_pos_acc))
            print(args.metric, ": current best val: {}, test: {}"
                  "current best val pos: {}, test: {}".format(best_valid_acc, best_test_acc, best_valid_pos_acc, best_test_pos_acc))
