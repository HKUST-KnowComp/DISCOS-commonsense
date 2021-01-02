ALL_SUBJS = [ "person", "man", "woman", 
             "someone", "somebody", "i", "he", "she", "you", ] 
SUBJ2POSS = {"i":"my", "he": "his", "she":"her", "you":"your"}

# Variables and rules
SUBJS = ["person", "man", "woman", 
        "someone", "somebody", "i", "he", "she", "you"]
O_SUBJS = ["i", "you", "he", "she"]
ATOMIC_SUBJS = ["PersonX", "PersonY", "PersonZ"] 

stative_rules = {
    "in":[], "out":[],
    "both_dir":["Synchronous", "Reason", "Result", "Condition", 
        "Conjunction", "Restatement", "Alternative"]
}
cause_agent_rules = {
    "out":["Succession", "Condition", "Reason", ],
    "in":["Precedence", "Result",], 
    "both_dir":["Synchronous", "Conjunction"], 
}
effect_agent_rules = {
    "out":["Precedence", "Result",], 
    "in":["Succession", "Condition", "Reason",],
    "both_dir":["Synchronous", "Conjunction"], 
}
# This requires, subj to be different
effect_theme_rules = {
    "out":["Precedence", "Result",], 
    "in":["Succession", "Condition", "Reason",],
    "both_dir":["Synchronous",  "Conjunction"], 
}

ASER_rules_dict = {
    "stative": stative_rules,
    "cause_agent": cause_agent_rules,
    "effect_agent": effect_agent_rules,
    "effect_theme": effect_theme_rules,
}

# functions:
def get_ppn_substitue_dict(head_split):
  """
      input (list): the split result of a head
      
      output: a dict tha maps personal pronouns in 
              head_split to subjects in ATOMIC_SUBJS
  """
  atomic_head_pp_list = []
  for token in head_split:
    if token in SUBJS:
      if not token in atomic_head_pp_list:
        atomic_head_pp_list.append(token)
  head_pp2atomic_pp = {}
  cnt = 0
  for pp in atomic_head_pp_list:
    head_pp2atomic_pp[pp] = ATOMIC_SUBJS[cnt]
    cnt += 1
    if cnt >= len(ATOMIC_SUBJS):
      break
  return head_pp2atomic_pp

def filter_event(event):
  """
      Function of filtering eventualities
      input (str): the string of eventuality
      
      output: whether to filter it out or not.
  """
  tokens = event.split()
#   if tokens[-1] in SUBJS and tokens[-2] == "tell":
#     return True
#   if tokens[-1] in ["know", "say", "think"]:
#     return True
  # filter eventualities with only 2 tokens
  if len(tokens) <= 2:
    return True
  # filter hot verbs
  if any(kw in tokens for kw in ["say", "do", "know", "tell", "think", ]):
    return True
  # filter out errors that potentially due to the errors of the parser
  if tokens[0] in ["who", "what", "when", "where", "how", "why", "which", "whom", "whose"]:
    return True
  return False  
