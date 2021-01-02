import math

def chunks_list(l, group_number):
  group_size = math.ceil(len(l) / group_number)
  final_data_groups = list()
  for i in range(0, len(l), group_size):
    final_data_groups.append(l[i:min(i+group_size, len(l))])
  return final_data_groups 