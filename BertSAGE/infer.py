import warnings
warnings.filterwarnings("ignore")
import torch
import os
from dataloader import *
from model import *
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='simple', type=str, required=False,
                    choices=["graphsage", "simple"],
                    help="choose model")
parser.add_argument("--model_path", default='', type=str, required=True,
                    help="model path")
parser.add_argument("--encoder", default='bert', type=str, required=False,
                    choices=["bert", "roberta"],
                    help="choose encoder")
parser.add_argument("--infer_path", default='', type=str, required=True,
                    help="npy file to be inferenced")
parser.add_argument("--graph_cach_path", default="graph_cache/.pickle", 
                    type=str, required=False,
                    help="path of graph cache")
parser.add_argument("--num_layers", default=1, type=int, required=False,
                    help="number of graphsage layers")
parser.add_argument("--num_neighbor_samples", default=4, type=int, required=False,
                    help="num neighbor samples in GraphSAGE")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_batch_size = 128

# graph_cache = args.graph_cach_path.format(
#     args.negative_sample, 
#     args.load_edge_types, 
#     os.path.basename(args.file_path).split(".")[0])
graph_cache = args.graph_cach_path

if args.model == "simple":
    with open(graph_cache, "rb") as reader:
        graph_dataset = pickle.load(reader)
    print("after loading graph cache from", graph_cache)
    data_loader = InferenceSimpleDataset(args.infer_path, device, args.encoder, graph_dataset)
elif args.model == "graphsage":
    with open(graph_cache, "rb") as reader:
        graph_dataset = pickle.load(reader)
    print("after loading graph cache from", graph_cache)
    data_loader = InferenceGraphDataset(args.infer_path, device, args.encoder, graph_dataset)

if args.model == "simple":
    model = SimpleClassifier(encoder=args.encoder,
                        adj_lists=None, 
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

model.load_state_dict(torch.load(args.model_path))
model.eval()

def infer(data_loader, model):
    def infer_mode(mode):
        all_predictions = []
        all_values = []
        all_hids = []
        for batch in tqdm(data_loader.get_batch(batch_size=test_batch_size, mode=mode)):
            b_s, _ = batch.shape # batch_size, 2+1
            all_nodes = batch[:, :2].reshape([-1])
            hids = batch[:, 2].tolist()

            logits = model(all_nodes, b_s) # (batch_size, 2)

            logits = torch.softmax(logits, dim=1)
            values = logits[:, 1]
            _, predicted = torch.max(logits, dim=1)
            predicted = predicted.tolist()
            values = values.tolist()
            all_predictions.extend(predicted)
            all_values.extend(values)
            all_hids.extend(hids)
        return all_predictions, all_values, all_hids

    with torch.no_grad():
        all_predictions = {}
        all_values = {}
        all_hids = {}
        for mode in ["head", "tail", "new"]:
            all_predictions[mode], all_values[mode], all_hids[mode] = infer_mode(mode)
    return all_predictions, all_values, all_hids

preds, vals, hids = infer(data_loader, model)
for mode in ["head", "tail", "new"]:
    print(mode, "num 1:", sum(np.array(preds[mode])==1), sum(np.array(preds[mode])==1)/len(preds[mode]))
# print the correct predict
plausible_knowledge = {}
for mode in ["head", "tail", "new"]:
    plausible_knowledge[mode] = []
    for i, (p, v, hid) in enumerate(zip(preds[mode], vals[mode], hids[mode])):
        plausible_knowledge[mode].append((data_loader.data[mode][i][:2], v, hid))
np.save("preds/"+os.path.basename(args.infer_path).split(".")[0]+"_preds"+"_"+args.model, plausible_knowledge)