import sys
sys.path.append('../')
import math
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor
from itertools import permutations, combinations_with_replacement, chain
from multiprocessing import Pool
from utils.atomic_utils import ALL_SUBJS, SUBJ2POSS

def instantiate_ppn(line):
    # slightly different from that in match_heads.py
    # 1. doesn't parse the sentence again
    # 2. return the original sentence if there's no PersonX/Y/Z
    strs = line.split()
    if len(strs) == 0:
        return []
#     strs = e_extractor.parse_text(line)
#     if len(strs) > 0:
#         strs = strs[0]['tokens']
#     else:
#         return []
    pp_index = []
    wildcard_index = []
    for i, word in enumerate(strs):
        if word in ["PersonX", "PersonY", "PersonZ"]:
            pp_index.append(i)
        # Deprecate replacing WILDCARD. This will be handled independently
#         elif word in ["WILDCARD", "something"]:
#             wildcard_index.append(i)
    # permutation of all possible substitutions
    perm_pp = list(combinations_with_replacement(ALL_SUBJS, len(pp_index)))
    perm_wildcard = list(combinations_with_replacement(['something', 'thing'], len(wildcard_index)))
    all_perms = [list(tmp_a)+list(tmp_b) for tmp_a in perm_pp for tmp_b in perm_wildcard]
    all_index = pp_index + wildcard_index
    if len(all_index) == 0:
        yield line
    else:
        modified_idx = []
        if "'s" in strs:
            for idx in pp_index:
                if strs[min(idx + 1, len(strs)-1)] == "'s":
                    modified_idx.append(idx)
        for perm in all_perms:
            # deal with possessive case
            if len(modified_idx) == 0:
                # if non of the PPs contain a following "'s", then just replace the heads
                yield ' '.join([strs[i] if not i in all_index else perm[all_index.index(i)]  for i in range(len(strs))])
            else:
                new_strs = [strs[i] if not i in all_index else perm[all_index.index(i)]  for i in range(len(strs))]
                for idx in modified_idx:
                    if new_strs[idx] in SUBJ2POSS:
                        new_strs[idx] = SUBJ2POSS[new_strs[idx]]
                        new_strs[idx+1] = "\REMOVE"
                while "\REMOVE" in new_strs:
                    new_strs.remove("\REMOVE")
                yield ' '.join(new_strs)

def contain_subject(dependencies):
    return any(dep in [item[1] for item in dependencies] for dep in ['nsubj', 'nsubjpass'])

def fill_sentence(sent, r, has_subject):
    if r in ['oEffect', 'xEffect']:
        # + subject
        if has_subject:
            return [sent]
        else:
            return [' '.join([subj, sent]) for subj in ALL_SUBJS]
    elif r in ['oReact', 'xReact']:
        # + subject / + subject is
        if has_subject:
            return [sent]
        else:
            return [' '.join([subj, sent]) for subj in ALL_SUBJS] + \
                    [' '.join([subj, 'is', sent]) for subj in ALL_SUBJS]
    elif r in ['xAttr']:
        # + subject is 
        if has_subject:
            return [sent]
        else:
            return [' '.join([subj, 'is', sent]) for subj in ALL_SUBJS]
    elif r in ['oWant', 'xWant']:
        # + subject want / + subject
        if has_subject:
            return [sent]
        else:
            # if start with 'to'
            if sent.lower().split()[0] == 'to':
                return [' '.join([subj, 'want', sent]) for subj in ALL_SUBJS] \
                             + [' '.join([subj, " ".join(sent.lower().split()[1:]) ]) for subj in ALL_SUBJS]
            else:
                return [' '.join([subj, 'want to', sent]) for subj in ALL_SUBJS] \
                             + [' '.join([subj, sent]) for subj in ALL_SUBJS]
    elif r in ['xIntent']:
        # + subject intent / + subject
        if has_subject:
            return [sent]
        else:
            # if start with 'to'
            if sent.lower().split()[0] == 'to':
                return [' '.join([subj, 'intent', sent]) for subj in ALL_SUBJS] \
                             + [' '.join([subj, " ".join(sent.lower().split()[1:]) ]) for subj in ALL_SUBJS]
            else:
                return [' '.join([subj, 'intent to', sent]) for subj in ALL_SUBJS]\
                             + [' '.join([subj, sent]) for subj in ALL_SUBJS]
    elif r in ['xNeed']:
        # + subject need / + subject
        if has_subject:
            return [sent]
        else:
            # if start with 'to'
            if sent.lower().split()[0] == 'to':
                return [' '.join([subj, 'need', sent]) for subj in ALL_SUBJS]\
                             + [' '.join([subj, " ".join(sent.lower().split()[1:]) ]) for subj in ALL_SUBJS]
            else:
                return [' '.join([subj, 'need to', sent]) for subj in ALL_SUBJS]\
                             + [' '.join([subj, sent]) for subj in ALL_SUBJS]

def unfold_parse_results(e):
    if len(e) == 0:
        return ""
    if len(e[0]) == 0:
        return ""
    return " ".join(e[0][0].words)        

def process_pp(sent):
    """
        Deal with the situation of "person x", "person y", "personx", "persony"
    """
    fill_words = {"person x":"PersonX", "person y":"PersonY", 
                  "personx":"PersonX", "persony":"PersonY",
                 "x":"PersonX", "y": "PersonY"}
    for strs in PP_filter_list:
        if strs in sent:
            sent = sent.replace(strs, fill_words[strs])
            break
    sent_split = sent.split()
    X_dict = {"X":"PersonX"}
    if "x" in sent_split or "y" in sent_split:
        sent = " ".join([fill_words.get(item, item) for item in sent_split])
    return sent


def extract(atomic_data, r, idx):
    extracted_event_list = [[] for i in range(len(atomic_data))]
    for i in tqdm(range(idx, len(atomic_data[r]), num_thread)): 
        tmp_node = []
        for sent in json.loads(atomic_data[r][i]):
            if sent == 'none':
                continue
            # filter the text
            sent = sent.lower()
            sent = process_pp(sent)            
            parsed_result = e_extractor.parse_text(sent)[0]
            filled_sentences = fill_sentence(sent, r, contain_subject(parsed_result['dependencies']))
            filled_sentences = list(chain(*[instantiate_ppn(s) for s in filled_sentences]))
            tmp_node.append([unfold_parse_results(e_extractor.extract_from_text(tmp_text))\
                               for tmp_text in filled_sentences])
        extracted_event_list[i] = tmp_node
        
    return extracted_event_list


parser = argparse.ArgumentParser()
parser.add_argument("--relation", default='xWant', type=str, required=True,
                    choices=['oEffect', 'oReact', 'oWant', 'xAttr', 
                             'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant'],
                    help="choose which relation to process")
parser.add_argument("--port", default=14000, type=int, required=False,
                    help="port of stanford parser")
args = parser.parse_args()

PP_filter_list = ["person x", "person y", "personx", "persony"]  

relation = args.relation

e_extractor = SeedRuleEventualityExtractor(
            corenlp_path = "stanford-corenlp-full/",
            corenlp_port= args.port)
atomic_data = pd.read_csv('v4_atomic_all_agg.csv')

num_thread = 5
workers = Pool(num_thread)
all_results = []
for i in range(num_thread):
    tmp_result = workers.apply_async(
        extract, 
        args=(atomic_data, relation, i))
    all_results.append(tmp_result)
    
workers.close()
workers.join()

all_results = [tmp_result.get() for tmp_result in all_results]
all_results = [list(chain(*item)) for item in zip(*all_results)]

np.save('ASER-format-words-final/ATOMIC_tails_'+relation, all_results)    