import sys
sys.path.append('../')
import math
import numpy as np
from tqdm import tqdm
from itertools import chain
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor
from itertools import permutations, combinations_with_replacement
from multiprocessing import Pool
from utils.atomic_utils import ALL_SUBJS, SUBJ2POSS

def instantiate_ppn(line):
    strs = e_extractor.parse_text(line)
    if len(strs) > 0:
        strs = strs[0]['tokens']
    else:
        return []
    pp_index = []
    wildcard_index = []
    for i, word in enumerate(strs):
        if word in ["PersonX", "PersonY", "PersonZ", "alex", "bob", "she", "he", "i", "you"]:
            pp_index.append(i)
        # Deprecate replacing WILDCARD. This will be handled independently
#         elif word in ["WILDCARD", "something"]:
#             wildcard_index.append(i)
    # permutation of all possible substitutions
    perm_pp = list(combinations_with_replacement(ALL_SUBJS, len(pp_index)))
    perm_wildcard = list(combinations_with_replacement(['something', 'thing'], len(wildcard_index)))
    all_perms = [list(tmp_a)+list(tmp_b) for tmp_a in perm_pp for tmp_b in perm_wildcard]
    all_index = pp_index + wildcard_index
    
    # deal with possesive cases
    modified_idx = []
    if "'s" in strs:
        for idx in pp_index:
            if strs[min(idx + 1, len(strs)-1)] == "'s":
                modified_idx.append(idx)
    for perm in all_perms:
        # deal with possessive case
        if len(modified_idx) == 0:
            # if none of the PPs contain a following "'s", then just replace the heads
            yield ' '.join([strs[i] if not i in all_index else perm[all_index.index(i)]  for i in range(len(strs))])
        else:
            # else, replace the PersonX's with my, her, his, etc.
            new_strs = [strs[i] if not i in all_index else perm[all_index.index(i)]  for i in range(len(strs))]
            for idx in modified_idx:
                if new_strs[idx] in SUBJ2POSS:
                    new_strs[idx] = SUBJ2POSS[new_strs[idx]]
                    new_strs[idx+1] = "\REMOVE"
            while "\REMOVE" in new_strs:
                new_strs.remove("\REMOVE")
            yield ' '.join(new_strs)

def unfold_parse_results(e):
    # return the words of the extractor results
    if len(e) == 0:
        return ""
    if len(e[0]) == 0:
        return ""
    return " ".join(e[0][0].words)

def extract(ATOMIC_lines, i):
    extracted_event_list = [[] for i in range(len(ATOMIC_lines))]
    for i in tqdm(range(i, len(ATOMIC_lines), num_thread)):
        line = ATOMIC_lines[i]
        possible_heads = instantiate_ppn(line)
        all_head_words = [unfold_parse_results(e_extractor.extract_from_text(tmp_text)) \
                       for tmp_text in possible_heads]
        extracted_event_list[i] = all_head_words
    return extracted_event_list

# main
stanford_patah = "stanford-corenlp-full/"
e_extractor = SeedRuleEventualityExtractor(
            corenlp_path = stanford_patah,
            corenlp_port= 13000)

ATOMIC_path = "all_agg_event.txt"
ATOMIC_lines = open().readlines()

num_thread = 5 
# the maximum number of a thread that the parser supports is 5
workers = Pool(num_thread)
all_results = []
for i in range(num_thread):
    tmp_result = workers.apply_async(
      extract, 
      args=(ATOMIC_lines, i))
    all_results.append(tmp_result)
    
workers.close()
workers.join()

all_results = [tmp_result.get() for tmp_result in all_results]
all_results = [list(chain(*item)) for item in zip(*all_results)]

np.save('ASER-format-words/ATOMIC_head_words_withpersonz', all_results)