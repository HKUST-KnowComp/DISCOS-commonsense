import os
import sys
import time
sys.path.append('../')
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor
from aser.database.kg_connection import ASERKGConnection

from utils.utils import chunks_list
from utils.atomic_utils import SUBJS, ATOMIC_SUBJS
from utils.atomic_utils import ASER_rules_dict

def get_all_cooccurances(relation_list):
  return sum([relation[1].get("Co_Occurrence", 0) for relation in relation_list])

def get_aser_neighbors_given_head(node, selected_head2tail_relations):
    """
        for a given ATOMIC head, 
        find selected one-hop neighbors by selected_head2tail_relations.
        
        for "in" and "out" relations, the retrieved aser_node can form:
          (head, aser_node)
        for "both_dir" relations, the retrieved aser_node can form:
          both (head, aser_node) and (aser_node, head)
    """
    
    # node as ASER head
    successor_dict = kg_conn.merged_eventuality_relation_cache["head_words"].get(node, {})
    selected_tails = [(key, get_all_cooccurances(relations))\
                        for key, relations in successor_dict.items()\
                       if any(any(r_name in selected_head2tail_relations["out"] for r_name in rs[1])\
                              for rs in relations)]
    
    # node as tail
    predecessor_dict = kg_conn.merged_eventuality_relation_cache["tail_words"].get(node, {})
    selected_heads = [(key, get_all_cooccurances(relations))\
                      for key, relations in predecessor_dict.items()\
                       if any(any(r_name in selected_head2tail_relations["in"] for r_name in rs[1])\
                              for rs in relations)]
    all_neighbor_dict = {**successor_dict, **predecessor_dict}
    
    selected_bothdir = [(key, get_all_cooccurances(relations))\
                        for key, relations in all_neighbor_dict.items()\
                       if any(any(r_name in selected_head2tail_relations["both_dir"] for r_name in rs[1])\
                              for rs in relations)]    
    # 
    return list(set(selected_tails + selected_heads)), list(set(selected_bothdir))
# 

def filter_event(event):
  tokens = event.split()
#   if tokens[-1] in SUBJS and tokens[-2] == "tell":
#     return True
#   if tokens[-1] in ["know", "say", "think"]:
#     return True
  if len(tokens) <= 2:
    return True
  if any(kw in tokens for kw in ["say", "do", "know", "tell", "think", ]):
    return True
  if tokens[0] in ["who", "what", "when", "where", "how", "why", "which", "whom", "whose"]:
    return True
  return False  

def extract(node_list):
  edges = []
  for node in tqdm(node_list):
    node_subj = node.split()[0]
    tails, both_dirs = get_aser_neighbors_given_head(node, selected_head2tail_relations)
    for tail, cooccur in tails:
      if tail in node_dict:
        if args.scenario != "effect_theme":
          # X-xx relation requires same subjects
          if tail.split()[0] == node_subj:
            edges.append((node_dict[node], node_dict[tail], {"cooccurance_time":cooccur}))
        else:
          # this relation requires different subjects
          if tail.split()[0] != node_subj:
              edges.append((node_dict[node], node_dict[tail], {"cooccurance_time":cooccur}))
        
    for tail, cooccur in both_dirs:
      if tail in node_dict:
        if args.scenario != "effect_theme":
          # X-xx relation requires same subjects
          if tail.split()[0] == node_subj:
            edges.append((node_dict[node], node_dict[tail], {"cooccurance_time":cooccur}))
            edges.append((node_dict[tail], node_dict[node], {"cooccurance_time":cooccur}))
        else:
          # this relation requires different subjects  
          if tail.split()[0] != node_subj:
            edges.append((node_dict[node], node_dict[tail], {"cooccurance_time":cooccur}))
            edges.append((node_dict[tail], node_dict[node], {"cooccurance_time":cooccur}))
  return edges
    
parser = argparse.ArgumentParser()
parser.add_argument("--scenario", default='', type=str, required=True,
                    choices=["stative","cause_agent","effect_agent","effect_theme"],
                    help="choose the ATOMIC scenario")
args = parser.parse_args()
selected_head2tail_relations = ASER_rules_dict[args.scenario]

st = time.time()
path_to_aser = "KG.db"
kg_conn = ASERKGConnection(path_to_aser, 
                           mode='memory', grain="words", load_types=["merged_eventuality", "words", "eventuality"])
print('time:', time.time()-st)

G_aser = nx.DiGraph()
for node in tqdm(kg_conn.merged_eventuality_cache):
  if filter_event(node):
    continue
  G_aser.add_node(node, freq=kg_conn.get_event_frequency(node),patterns=kg_conn.get_event_patterns(node))

node_dict = dict([(node, i) for i, node in enumerate(G_aser.nodes())])
node_feat_dict = dict(G_aser.nodes.data())    
if not os.path.exists("aser-graph-file/ASER_core_node2id.npy"):
    np.save("aser-graph-file/ASER_core_node2id", node_dict)

number_of_worker = 25

node_list_chunks = chunks_list(list(G_aser.nodes()), number_of_worker)

workers = Pool(number_of_worker)
all_results = list()
for i in range(number_of_worker):
    tmp_result = workers.apply_async(
      extract, 
      args=(node_list_chunks[i], ))
    all_results.append(tmp_result)
    
workers.close()
workers.join()

G_new = nx.DiGraph()
for node in G_aser:
  G_new.add_node(node_dict[node], freq=node_feat_dict[node]["freq"], patterns=node_feat_dict[node]["patterns"])
for edges in all_results:
  G_new.add_edges_from(edges.get())
nx.write_gpickle(G_new, "aser-graph-file/G_aser_{}.pickle".format(args.scenario))