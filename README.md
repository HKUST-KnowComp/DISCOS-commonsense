# DISCOS-commonsense

This is the github repo for The Web Conference (WWW) 2021 paper [DISCOS: Bridging the Gap between Discourse Knowledge and Commonsense Knowledge](https://arxiv.org/abs/2101.00154).

### How to train the Commonsense Knowledge Graph Population (CKGP) model

Here is the instruction for learning the CKGP model. We use the filtered graph as introduced in Section 5.1.1 for this experiment.

First, git clone this repo, and then download the prepared aligned graph from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EqYM_lq9gl1DhJu6HnezBvYBzuOfk60iDhg_zCTq9gZrLw?e=18OxwY). The `data/graph_cache` folder contains the data that can be directly used for training and testing. The `data/graph_raw_data` is the graph file that we got after aligning ATOMIC and ASER, with pre-defined negative edges for the CKGP task. The `data/infer_candidates` folder contains the candidate (h, r, t) tuples to be scored by our BertSAGE model.

Next install dependencies. Recommended python version is 3.8+.

`pip install -r requirements.txt`

Note that to load the files in `data/graph_cache` requires the same dependencies as in the `requirements.txt` file. E.g., you need to install the `transformers` package with version 3.4.0.

Next train the `BertSAGE` model. For example here is the command to train with the `oReact` relation:

```
python -u BertSAGE/train.py --model graphsage \
    --load_edge_types ASER \
    --neg_prop 1 \
    --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle \
    --negative_sample prepared_neg \
    --file_path data/graph_raw_data/G_aser_oReact_1hop_thresh_100_neg_other_20_inv_10.pickle
```

For other relations, you could find the corresponding `.pickle` file from `data` folder.

For the inference part, you could run after training:

```
python -u BertSAGE/infer.py --gpu 0 --model graphsage \
    --model_path models/G_aser_oReact_1hop_thresh_100_neg_other_20_inv_10/graphsage_best_bert_bs64_opt_SGD_lr0.01_decay0.8_500_layer1_neighnum_4_graph_ASER_acc.pth \
    --infer_path data/infer_candidates/G_aser_oReact_1hop_thresh_100_neg_other_20_inv_10.npy \
    --graph_cach_path data/graph_cache/neg_prepared_neg_ASER_G_aser_oReact_1hop_thresh_100_neg_other_20_inv_10.pickle
```


### The Acquired Knowledge Graph DISCOS-ATOMIC

By populating the knowledge in ATOMIC to the whole ASER, we can acquire a large-scale ATOMIC-like knowledge graph by selecting the tuples scored by BertSAGE over 0.5. Also, we present the acquisition results of DISCOS under the setting of COMET, i.e., given h and r to generate t. The new knowledge graph can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/ElHMMtHsCwZLg-AdP8ZdJT8BCBwTOyAOil1XLt4EfPYWUg?e=49u0i3).

The 3.4M if-then knowledge is populated using the whole graph of ASER-core, without the neighbor filtering. You may find the processed training graph and inference candidates [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EmC5tdRCmQlMrfwBHrVHYE4B5_UhIfqL1uxNSNofLPMYQQ?e=dkQObG). 






