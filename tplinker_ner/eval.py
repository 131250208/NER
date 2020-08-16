#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
import os
config = yaml.load(open("eval_config.yaml", "r"), Loader = yaml.FullLoader)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
import json
from tqdm import tqdm
import re
from IPython.core.debugger import set_trace
from pprint import pprint
import unicodedata
from transformers import AutoModel, BasicTokenizer, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
from ner_common.utils import Preprocessor, WordTokenizer
from tplinker_ner import (HandshakingTaggingScheme,
                          DataMaker, 
                          TPLinkerNER,
                          Metrics)
import wandb
import numpy as np
from collections import OrderedDict


# In[4]:


print(torch.cuda.device_count(), "GPUs are available")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# meta_path = os.path.join(data_home, exp_name, config["meta"])
# batch_size = config["batch_size"]
# encoder_path = config["bert_path"]
# visual_field = config["visual_field"]
# use_last_k_layers_hiddens = config["use_last_k_layers_hiddens"]
# add_bilstm_on_the_top = config["add_bilstm_on_the_top"]
# bilstm_layers = config["bilstm_layers"]
# bilstm_dropout = config["bilstm_dropout"]

# >>>>>>>>>data path>>>>>>>>>>>>>>


# In[ ]:


# for reproductivity
torch.backends.cudnn.deterministic = True

# >>>>>>>>predication config>>>>>>
max_seq_len = config["max_seq_len"]
sliding_len = config["sliding_len"]
batch_size = config["batch_size"]

# >>>>>>>>>>model config>>>>>>>>>>>>
## char encoder
char_encoder_config = config["char_encoder_config"] if config["use_char_encoder"] else None

## bert encoder
bert_config = config["bert_config"] if config["use_bert"] else None
use_bert = config["use_bert"]

## word encoder
word_encoder_config = config["word_encoder_config"] if config["use_word_encoder"] else None

## flair config
flair_config = {
    "embedding_ids": config["flair_embedding_ids"],
} if config["use_flair"] else None

## handshaking_kernel
handshaking_kernel_config = config["handshaking_kernel_config"]
visual_field = handshaking_kernel_config["visual_field"]

## encoding fc
enc_hidden_size = config["enc_hidden_size"]
activate_enc_fc = config["activate_enc_fc"]

# >>>>>>>>>data path>>>>>>>>>>>>>>
data_home = config["data_home"]
exp_name = config["exp_name"]
meta_path = os.path.join(data_home, exp_name, config["meta"])
save_res_dir = os.path.join(config["save_res_dir"], exp_name)
test_data_path = os.path.join(data_home, exp_name, config["test_data"])
word2idx_path = os.path.join(data_home, exp_name, config["word2idx"])
char2idx_path = os.path.join(data_home, exp_name, config["char2idx"])


# In[5]:


test_data_path_dict = {}
for path in glob.glob(test_data_path):
    file_name = re.search("(.*?)\.json", path.split("/")[-1]).group(1)
    test_data_path_dict[file_name] = path


# # Load Data

# In[6]:


test_data_dict = {}
for file_name, path in test_data_path_dict.items():
    test_data_dict[file_name] = json.load(open(path, "r", encoding = "utf-8"))


# # Split

# In[7]:


# init tokenizers
if use_bert:
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_config["path"], add_special_tokens = False, do_lower_case = False)
word2idx = json.load(open(word2idx_path, "r", encoding = "utf-8"))
word_tokenizer = WordTokenizer(word2idx)

# preprocessor
tokenizer4preprocess = bert_tokenizer if use_bert else word_tokenizer
preprocessor = Preprocessor(tokenizer4preprocess, use_bert)


# In[ ]:


def split(data, max_seq_len, sliding_len, data_name = "train"):
    '''
    split into short texts
    '''
    max_tok_num = 0
    for sample in tqdm(data, "calculating the max token number of {}".format(data_name)):
        text = sample["text"]
        tokens = preprocessor.tokenize(text)
        max_tok_num = max(max_tok_num, len(tokens))
    print("max token number of {}: {}".format(data_name, max_tok_num))
    
    if max_tok_num > max_seq_len:
        print("max token number of {} is greater than the setting, need to split!, max_seq_len of {} is {}".format(data_name, data_name, max_seq_len))
        short_data = preprocessor.split_into_short_samples(data, 
                                          max_seq_len, 
                                          sliding_len = sliding_len, 
                                          data_type = "test")
    else:
        short_data = data
        max_seq_len = max_tok_num
        print("max token number of {} is less than the setting, no need to split! max_seq_len of {} is reset to {}.".format(data_name, data_name, max_tok_num))
    return short_data, max_seq_len


# In[9]:


# all_data = []
# for data in list(test_data_dict.values()):
#     all_data.extend(data)
    
# max_tok_num = 0
# for sample in tqdm(all_data, desc = "Calculate the max token number"):
#     tokens = tokenize(sample["text"])
#     max_tok_num = max(len(tokens), max_tok_num)


# In[10]:


# split_test_data = False
# if max_tok_num > config["max_test_seq_len"]:
#     split_test_data = True
#     print("max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(max_tok_num, config["max_test_seq_len"]))
# else:
#     print("max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(max_tok_num, config["max_test_seq_len"]))
# max_seq_len = min(max_tok_num, config["max_test_seq_len"]) 

# if config["force_split"]:
#     split_test_data = True
#     print("force to split the test dataset!")    

ori_test_data_dict = copy.deepcopy(test_data_dict)
test_data_dict = {}
max_seq_len_all_data = []
for file_name, data in ori_test_data_dict.items():
    split_data, max_seq_len_this_data = split(data, max_seq_len, sliding_len, file_name)
    max_seq_len_all_data.append(max_seq_len_this_data)
    test_data_dict[file_name] = split_data
max_seq_len = max(max_seq_len_all_data)
print("max_seq_len is reset to {}.".format(max_seq_len))


# In[11]:


for filename, short_data in test_data_dict.items():
    print("{}: {}".format(filename, len(short_data)))


# # Decoder(Tagger)

# In[12]:


meta = json.load(open(meta_path, "r", encoding = "utf-8"))
tags = meta["tags"]
if meta["visual_field_rec"] > visual_field:
    visual_field = meta["visual_field_rec"]
    print("Recommended visual_field is greater than current visual_field, reset to rec val: {}".format(visual_field))


# In[13]:


handshaking_tagger = HandshakingTaggingScheme(tags, max_seq_len, handshaking_kernel_config["visual_field"])


# # Character indexing

# In[ ]:


char2idx = json.load(open(char2idx_path, "r", encoding = "utf-8"))
def text2char_indices(text, max_seq_len = -1):
    char_ids = []
    chars = list(text)
    for c in chars:
        if c not in char2idx:
            char_ids.append(char2idx['<UNK>'])
        else:
            char_ids.append(char2idx[c])
    if len(char_ids) < max_seq_len:
        char_ids.extend([char2idx['<PAD>']] * (max_seq_len - len(char_ids)))
    if max_seq_len != -1:
        char_ids = torch.tensor(char_ids[:max_seq_len]).long()
    return char_ids


# # Dataset

# In[14]:


if use_bert:
    data_maker = DataMaker(handshaking_tagger, word_tokenizer, text2char_indices, bert_tokenizer)
else:
    data_maker = DataMaker(handshaking_tagger, word_tokenizer, text2char_indices)


# In[15]:


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


# In[ ]:


# max word num, max subword num, max char num
def cal_max_tok_num(data, tokenizer):
    max_tok_num = 0
    for example in data:
        text = example["text"]
        max_tok_num = max(max_tok_num, len(tokenizer.tokenize(text)))
    return max_tok_num


# In[ ]:


# max character num of a single word
def get_max_char_num_in_subword(data):
    max_char_num = 0
    for example in data:
        text = example["text"]
        subword2char_span = bert_tokenizer.encode_plus(text, 
                                                       return_offsets_mapping = True, 
                                                       add_special_tokens = False)["offset_mapping"]
        max_char_num = max([span[1] - span[0] for span in subword2char_span] + [max_char_num, ])
    return max_char_num

def get_max_char_num_in_word(data):
    max_char_num = 0
    for example in data:
        text = example["text"]
        word2char_span = word_tokenizer.encode_plus(text)["offset_mapping"]
        max_char_num = max([span[1] - span[0] for span in word2char_span] + [max_char_num, ])
    return max_char_num


# In[ ]:


all_data = []
for data in list(test_data_dict.values()):
    all_data.extend(data)

# 
max_word_num = cal_max_tok_num(all_data, word_tokenizer)
print("max_word_num: {}".format(max_word_num))
if use_bert:
    max_subword_num = cal_max_tok_num(all_data, bert_tokenizer)
    print("max_subword_num: {}".format(max_subword_num))

# max_char_num_in_tok   
if use_bert:
    max_char_num_in_tok = get_max_char_num_in_subword(all_data)
else:
    max_char_num_in_tok = get_max_char_num_in_word(all_data)
print("max_char_num_in_tok: {}".format(max_char_num_in_tok))


# # Model

# In[16]:


if char_encoder_config is not None:
    char_encoder_config["char_size"] = len(char2idx)
    char_encoder_config["max_char_num_in_tok"] = max_char_num_in_tok
if word_encoder_config is not None:
    word_encoder_config["word2idx"] = word2idx
ent_extractor = TPLinkerNER(char_encoder_config,
                            word_encoder_config,
                            flair_config,
                            handshaking_kernel_config,
                            enc_hidden_size,
                            activate_enc_fc,
                            len(tags),
                            bert_config,
                            )
ent_extractor = ent_extractor.to(device)


# # Merics

# In[17]:


metrics = Metrics(handshaking_tagger)


# # Prediction

# In[18]:


# get model state paths
model_state_dir = config["model_state_dict_dir"]
target_run_ids = set(config["run_ids"])
run_id2model_state_paths = {}
for root, dirs, files in os.walk(model_state_dir):
    for file_name in files:
        run_id = root.split("-")[-1]
        if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
            if run_id not in run_id2model_state_paths:
                run_id2model_state_paths[run_id] = []
            model_state_path = os.path.join(root, file_name)
            run_id2model_state_paths[run_id].append(model_state_path)
print("Following model states will be loaded: ")
pprint(run_id2model_state_paths)


# In[19]:


def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key = lambda x: int(re.search("(\d+)", x.split("/")[-1]).group(1)))
#     pprint(path_list)
    return path_list[-k:]


# In[20]:


# only last k models
k = config["last_k_model"]
for run_id, path_list in run_id2model_state_paths.items():
    run_id2model_state_paths[run_id] = get_last_k_paths(path_list, k)


# In[21]:


def filter_duplicates(ent_list):
    ent_memory_set = set()
    filtered_ent_list = []
    for ent in ent_list:
        ent_memory = "{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"])
        if ent_memory not in ent_memory_set:
            filtered_ent_list.append(ent)
            ent_memory_set.add(ent_memory)
    return filtered_ent_list


# In[22]:


def predict(test_data, ori_test_data):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
#     indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, data_type = "test") # fill up to max_seq_len
    if use_bert:
        indexed_test_data = data_maker.get_indexed_data(test_data,
                                                                max_word_num,
                                                                max_char_num_in_tok, 
                                                                max_subword_num_train, 
                                                                data_type = "test")
    else:
        indexed_test_data = data_maker.get_indexed_data(test_data,
                                                        max_word_num,
                                                        max_char_num_in_tok, 
                                                        data_type = "test")
    test_dataloader = DataLoader(MyDataset(indexed_test_data), 
                              batch_size = batch_size, 
                              shuffle = False, 
                              num_workers = 6,
                              drop_last = False,
                              collate_fn = lambda data_batch: data_maker.generate_batch(data_batch, use_bert = use_bert, data_type = "test"),
                             )
    
    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc = "Predicting"):
#         if config["encoder"] == "BERT":
#             sample_list, batch_input_ids, \
#             batch_attention_mask, batch_token_type_ids, \
#             tok2char_span_list, _ = batch_test_data

#             batch_input_ids, \
#             batch_attention_mask, \
#             batch_token_type_ids = (batch_input_ids.to(device), 
#                                       batch_attention_mask.to(device), 
#                                       batch_token_type_ids.to(device))

# #         elif config["encoder"] in {"BiLSTM", }:
# #             text_id_list, text_list, batch_input_ids, tok2char_span_list = batch_test_data
# #             batch_input_ids = batch_input_ids.to(device)
     
#         with torch.no_grad():
#             if config["encoder"] == "BERT":
#                 batch_pred_shaking_outputs = ent_extractor(batch_input_ids, 
#                                                           batch_attention_mask, 
#                                                           batch_token_type_ids, 
#                                                          )
# #             elif config["encoder"] in {"BiLSTM", }:
# #                 batch_pred_shaking_outputs = ent_extractor(batch_input_ids)
        if bert_config is not None:
            sample_list,             padded_sent_list,             batch_subword_input_ids,             batch_attention_mask,             batch_token_type_ids,             tok2char_span_list,             batch_char_input_ids4subword,             batch_word_input_ids,             batch_subword2word_idx_map,             batch_gold_shaking_tag = batch_test_data
        else:
            sample_list,             padded_sent_list,             batch_char_input_ids4subword,             batch_word_input_ids,             tok2char_span_list,             batch_gold_shaking_tag = batch_test_data

        batch_char_input_ids4subword,         batch_word_input_ids,         batch_gold_shaking_tag = (batch_char_input_ids4subword.to(device), 
                                  batch_word_input_ids.to(device),
                                  batch_gold_shaking_tag.to(device) 
                                     )
        if bert_config is not None:
                batch_subword_input_ids,                 batch_attention_mask,                 batch_token_type_ids,                 batch_subword2word_idx_map = (batch_subword_input_ids.to(device), 
                                              batch_attention_mask.to(device), 
                                              batch_token_type_ids.to(device), 
                                              batch_subword2word_idx_map.to(device))

        with torch.no_grad():
            if bert_config is not None:
                batch_pred_shaking_outputs = ent_extractor(batch_char_input_ids4subword, 
                                                           batch_word_input_ids,
                                                           padded_sent_list,
                                                           batch_subword_input_ids, 
                                                           batch_attention_mask, 
                                                           batch_token_type_ids, 
                                                           batch_subword2word_idx_map)
            else:
                batch_pred_shaking_outputs = ent_extractor(batch_char_input_ids4subword, 
                                                           batch_word_input_ids,
                                                           padded_sent_list
                                                           )
        batch_pred_shaking_tag = (batch_pred_shaking_outputs > 0.).long()

        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            text_id = sample["id"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]
            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = sample["tok_offset"], sample["char_offset"]
            ent_list = handshaking_tagger.decode_ent(text, 
                                                     pred_shaking_tag, 
                                                     tok2char_span, 
                                                     tok_offset = tok_offset, 
                                                     char_offset = char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "entity_list": ent_list,
            })
            
    # merge
    text_id2ent_list = {}
    for sample in pred_sample_list:
        text_id = sample["id"]
        if text_id not in text_id2ent_list:
            text_id2ent_list[text_id] = sample["entity_list"]
        else:
            text_id2ent_list[text_id].extend(sample["entity_list"])

    text_id2text = {sample["id"]:sample["text"] for sample in ori_test_data}
    merged_pred_sample_list = []
    for text_id, ent_list in text_id2ent_list.items():
        merged_pred_sample_list.append({
            "id": text_id,
            "text": text_id2text[text_id],
            "entity_list": filter_duplicates(ent_list),
        })
        
    return merged_pred_sample_list


# In[23]:


def get_test_prf(pred_sample_list, gold_test_data, pattern = "only_head"):
    text_id2gold_n_pred = {}
    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_entity_list": sample["entity_list"],
        }
    
    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_entity_list"] = sample["entity_list"]

    correct_num, pred_num, gold_num = 0, 0, 0
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_ent_list = gold_n_pred["gold_entity_list"]
        pred_ent_list = gold_n_pred["pred_entity_list"] if "pred_entity_list" in gold_n_pred else []
        if pattern == "only_head_index":
            gold_ent_set = set(["{}\u2E80{}".format(ent["char_span"][0], ent["type"]) for ent in gold_ent_list])
            pred_ent_set = set(["{}\u2E80{}".format(ent["char_span"][0], ent["type"]) for ent in pred_ent_list])
        elif pattern == "whole_span":
            gold_ent_set = set(["{}\u2E80{}\u2E80{}".format(ent["char_span"][0], ent["char_span"][1], ent["type"]) for ent in gold_ent_list])
            pred_ent_set = set(["{}\u2E80{}\u2E80{}".format(ent["char_span"][0], ent["char_span"][1], ent["type"]) for ent in pred_ent_list])
        elif pattern == "whole_text":
            gold_ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in gold_ent_list])
            pred_ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in pred_ent_list])
            
        for ent_str in pred_ent_set:
            if ent_str in gold_ent_set:
                correct_num += 1

        pred_num += len(pred_ent_set)
        gold_num += len(gold_ent_set)
#     print((correct_num, pred_num, gold_num))
    prf = metrics.get_scores(correct_num, pred_num, gold_num)
    return prf


# In[24]:


# predict
res_dict = {}
predict_statistics = {}
for file_name, short_data in test_data_dict.items():
    ori_test_data = ori_test_data_dict[file_name]
    for run_id, model_path_list in run_id2model_state_paths.items():
        save_dir4run = os.path.join(save_res_dir, run_id)
        if config["save_res"] and not os.path.exists(save_dir4run):
            os.makedirs(save_dir4run)
            
        for model_state_path in model_path_list:
            res_num = re.search("(\d+)", model_state_path.split("/")[-1]).group(1)
            save_path = os.path.join(save_dir4run, "{}_res_{}.json".format(file_name, res_num))
            
            if os.path.exists(save_path):
                pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding = "utf-8")]
                print("{} already exists, load it directly!".format(save_path))
            else:
                # load model state
                model_state_dict = torch.load(model_state_path)
                # if used paralell train, need to rm prefix "module."
                new_model_state_dict = OrderedDict()
                for key, v in model_state_dict.items():
                    key = re.sub("module\.", "", key)
                    new_model_state_dict[key] = v
                ent_extractor.load_state_dict(new_model_state_dict)
                ent_extractor.eval()
                print("run_id: {}, model state {} loaded".format(run_id, model_state_path.split("/")[-1]))

                # predict
                pred_sample_list = predict(short_data, ori_test_data)
            
            res_dict[save_path] = pred_sample_list
            predict_statistics[save_path] = len([s for s in pred_sample_list if len(s["entity_list"]) > 0])
pprint(predict_statistics)


# In[25]:


# score
if config["score"]:
    filepath2scores = {}
    for file_path, pred_samples in res_dict.items():
        file_name = re.match("(.*?)_res_\d+.json", file_path.split("/")[-1]).group(1)
        gold_test_data = ori_test_data_dict[file_name]
        prf = get_test_prf(pred_samples, gold_test_data, pattern = config["correct"])
        filepath2scores[file_path] = prf
    print("---------------- Results -----------------------")
    pprint(filepath2scores)


# In[26]:


# check char span
for path, res in res_dict.items():
    for sample in tqdm(res, "check character level span"):
        text = sample["text"]
        for ent in sample["entity_list"]:
            assert ent["text"] == text[ent["char_span"][0]:ent["char_span"][1]]


# In[27]:


# save 
if config["save_res"]:
    for path, res in res_dict.items():
        with open(path, "w", encoding = "utf-8") as file_out:
            for sample in tqdm(res, desc = "Output"):
                if len(sample["entity_list"]) == 0:
                    continue
                json_line = json.dumps(sample, ensure_ascii = False)     
                file_out.write("{}\n".format(json_line))

