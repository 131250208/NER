#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
import os
config = yaml.load(open("train_config.yaml", "r"), Loader = yaml.FullLoader)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
from bs4 import BeautifulSoup
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import BertTokenizerFast
from IPython.core.debugger import set_trace
import copy
from tqdm import tqdm
import html
from pprint import pprint
import glob
import time
from ner_common.utils import Preprocessor, WordTokenizer
from tplinker_ner import (HandshakingTaggingScheme, 
                          DataMaker, 
                          TPLinkerNER, 
                          Metrics)
import json
import wandb
import numpy as np


# # Superparameter

# In[ ]:


print(torch.cuda.device_count(), "GPUs are available")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# hyperparameters
hyper_parameters = config["hyper_parameters"]
experiment_name = config["experiment_name"]
run_name = config["run_name"]
if config["wandb"] == True:
    # init wandb
    wandb.init(project = experiment_name, 
               name = run_name,
               config = hyper_parameters # Initialize config
              )

    wandb.config.note = config["note"]          

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    model_state_dict_dir = os.path.join(config["path_to_save_model"], experiment_name)
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)


# In[ ]:


# >>>>>>>>>>>preprocessing config>>>>>>>>>>>>>>
max_seq_len = hyper_parameters["max_seq_len"]
pred_max_seq_len = hyper_parameters["pred_max_seq_len"]
sliding_len = hyper_parameters["sliding_len"]
pred_sliding_len = hyper_parameters["pred_sliding_len"]

# >>>>>>>>>>>>>train config>>>>>>>>>>
## for reproductivity
seed = hyper_parameters["seed"]
torch.manual_seed(seed) # pytorch random seed
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
init_learning_rate = float(hyper_parameters["lr"])
batch_size = hyper_parameters["batch_size"]
parallel = hyper_parameters["parallel"]
if parallel:
    print("Parallel training is set up!")
epoch_num = hyper_parameters["epochs"]

# >>>>>>>>>>model config>>>>>>>>>>>>
## char encoder
char_encoder_config = hyper_parameters["char_encoder_config"] if hyper_parameters["use_char_encoder"] else None

## bert encoder
bert_config = hyper_parameters["bert_config"] if hyper_parameters["use_bert"] else None
use_bert = hyper_parameters["use_bert"]

## word encoder
word_encoder_config = hyper_parameters["word_encoder_config"] if hyper_parameters["use_word_encoder"] else None

## flair config
flair_config = {
    "embedding_ids": hyper_parameters["flair_embedding_ids"],
} if hyper_parameters["use_flair"] else None

## handshaking_kernel
handshaking_kernel_config = hyper_parameters["handshaking_kernel_config"]

## encoding fc
enc_hidden_size = hyper_parameters["enc_hidden_size"]
activate_enc_fc = hyper_parameters["activate_enc_fc"]


# >>>>>>>>>data path>>>>>>>>>>>>>>
data_home = config["data_home"]
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
meta_path = os.path.join(data_home, experiment_name, config["meta"])
word2idx_path = os.path.join(data_home, experiment_name, config["word2idx"])
char2idx_path = os.path.join(data_home, experiment_name, config["char2idx"])


# # Load Data

# In[ ]:


train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))


# # Split

# In[ ]:


# init tokenizers
if use_bert:
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_config["path"], add_special_tokens = False, do_lower_case = False)
word2idx = json.load(open(word2idx_path, "r", encoding = "utf-8"))
word_tokenizer = WordTokenizer(word2idx)

tokenizer4preprocess = bert_tokenizer if use_bert else word_tokenizer
preprocessor = Preprocessor(tokenizer4preprocess, use_bert)


# In[ ]:


def split(data, max_seq_len, sliding_len, data_type = "train"):
    '''
    split into short texts
    '''
    max_tok_num = 0
    for sample in tqdm(data, "calculating the max token number of {} data".format(data_type)):
        text = sample["text"]
        tokens = preprocessor.tokenize(text)
        max_tok_num = max(max_tok_num, len(tokens))
    print("max token number of {} data: {}".format(data_type, max_tok_num))
    
    if max_tok_num > max_seq_len:
        print("max token number of {} data is greater than the setting, need to split!".format(data_type))
        short_data = preprocessor.split_into_short_samples(data, 
                                          max_seq_len, 
                                          sliding_len = sliding_len)
    else:
        short_data = data
        print("max token number of {} data is less than the setting, no need to split!".format(data_type))
    return short_data
short_train_data = split(train_data, max_seq_len, sliding_len, "train")
short_valid_data = split(valid_data, pred_max_seq_len, pred_sliding_len, "valid")


# In[ ]:


# # check tok spans of new short article dict list
# for art_dict in tqdm(short_train_data + short_valid_data):
#     text = art_dict["text"]
#     tok2char_span = preprocessor.get_tok2char_span_map(text)
#     for term in art_dict["entity_list"]:        
#         # # bert-base-cased 的voc 里必须加两个token：hypo, mineralo
#         tok_span = term["tok_span"]
#         char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
#         pred_text = text[char_span_list[0][0]:char_span_list[-1][1]]
#         assert pred_text == term["text"]


# # Tagging

# In[ ]:


meta = json.load(open(meta_path, "r", encoding = "utf-8"))
tags = meta["tags"]
if meta["visual_field_rec"] > handshaking_kernel_config["visual_field"]:
    handshaking_kernel_config["visual_field"] = meta["visual_field_rec"]
    print("Recommended visual_field is greater than current visual_field, reset to rec val: {}".format(handshaking_kernel_config["visual_field"]))


# In[ ]:


def sample_equal_to(sample1, sample2):
    assert sample1["id"] == sample2["id"] and sample1["text"] == sample2["text"]
    
    entity_list1, entity_list2 = sample1["entity_list"], sample2["entity_list"]
    memory_set = set()
    for term in entity_list2:
        memory_set.add("{},{},{}".format(term["tok_span"][0], term["tok_span"][1], term["type"]))
    for term in entity_list1:
        memory = "{},{},{}".format(term["tok_span"][0], term["tok_span"][1], term["type"])
        if memory not in memory_set:
            set_trace()
            return False
    return True


# In[ ]:


handshaking_tagger = HandshakingTaggingScheme(tags, max_seq_len, handshaking_kernel_config["visual_field"])
handshaking_tagger4valid = HandshakingTaggingScheme(tags, pred_max_seq_len, handshaking_kernel_config["visual_field"])


# In[ ]:


# # check tagging and decoding
# def check_tagging_decoding(data4check, handshaking_tagger):
#     for idx in tqdm(range(0, len(data4check), batch_size)):
#         batch_matrix_spots = []
#         batch_data = data4check[idx:idx + batch_size]
#         for sample in batch_data:
#             matrix_spots = handshaking_tagger.get_spots(sample)
#     #         %timeit shaking_tagger.get_spots(sample)
#             batch_matrix_spots.append(matrix_spots)

#         # tagging
#         # batch_shaking_tag: (batch_size, shaking_tag, tag_size)
#         batch_shaking_tag = handshaking_tagger.spots2shaking_tag4batch(batch_matrix_spots)
#     #     %timeit shaking_tagger.spots2shaking_tag4batch(batch_matrix_spots) #0.3s

#         for batch_idx in range(len(batch_data)):
#             gold_sample = batch_data[batch_idx]
#             shaking_tag = batch_shaking_tag[batch_idx]
#             # decode
#             text = gold_sample["text"]
#             tok2char_span = get_tok2char_span_map(text)
#             ent_list = handshaking_tagger.decode_ent(text, shaking_tag, tok2char_span)
#             pred_sample = {
#                 "text": text,
#                 "id": gold_sample["id"],
#                 "entity_list": ent_list,
#             }

#             if not sample_equal_to(pred_sample, gold_sample) or not sample_equal_to(gold_sample, pred_sample):
#                 set_trace()
# check_tagging_decoding(short_train_data, handshaking_tagger)
# check_tagging_decoding(short_valid_data, handshaking_tagger4valid)


# # Character Index

# In[ ]:


# character indexing
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

# In[ ]:


if use_bert:
    data_maker = DataMaker(handshaking_tagger, word_tokenizer, text2char_indices, bert_tokenizer)
    data_maker4valid = DataMaker(handshaking_tagger4valid, word_tokenizer, text2char_indices, bert_tokenizer)
else:
    data_maker = DataMaker(handshaking_tagger, word_tokenizer, text2char_indices)
    data_maker4valid = DataMaker(handshaking_tagger4valid, word_tokenizer, text2char_indices)


# In[ ]:


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

max_word_num_train = cal_max_tok_num(short_train_data, word_tokenizer)
max_word_num_valid = cal_max_tok_num(short_valid_data, word_tokenizer)
print("max_word_num_train: {}, max_word_num_val: {}".format(max_word_num_train, max_word_num_valid))
if use_bert:
    max_subword_num_train = cal_max_tok_num(short_train_data, bert_tokenizer)
    max_subword_num_valid = cal_max_tok_num(short_valid_data, bert_tokenizer)
    print("max_subword_num_train: {}, max_subword_num_val: {}".format(max_subword_num_train, max_subword_num_valid))


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


if use_bert:
    max_char_num_in_tok = get_max_char_num_in_subword(short_train_data + short_valid_data)
else:
    max_char_num_in_tok = get_max_char_num_in_word(short_train_data + short_valid_data)


# In[ ]:


if use_bert:
    indexed_train_sample_list = data_maker.get_indexed_data(short_train_data,
                                                            max_word_num_train,
                                                            max_char_num_in_tok, 
                                                            max_subword_num_train)
    indexed_valid_sample_list = data_maker4valid.get_indexed_data(short_valid_data,
                                                                  max_word_num_valid,
                                                                  max_char_num_in_tok, 
                                                                  max_subword_num_valid)
else:
    indexed_train_sample_list = data_maker.get_indexed_data(short_train_data,
                                                            max_word_num_train,
                                                            max_char_num_in_tok)
    indexed_valid_sample_list = data_maker4valid.get_indexed_data(short_valid_data,
                                                                  max_word_num_valid,
                                                                  max_char_num_in_tok)


# In[ ]:



train_dataloader = DataLoader(MyDataset(indexed_train_sample_list), 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  num_workers = 6,
                                  drop_last = False,
                                  collate_fn = lambda batch_data: data_maker.generate_batch(batch_data, use_bert = use_bert),
                                 )
valid_dataloader = DataLoader(MyDataset(indexed_valid_sample_list), 
                          batch_size = batch_size, 
                          shuffle = False, 
                          num_workers = 6,
                          drop_last = False,
                          collate_fn = lambda batch_data: data_maker4valid.generate_batch(batch_data, use_bert = use_bert),
                         )


# In[ ]:


# # have a look at dataloader
# train_data_iter = iter(train_dataloader)
# batch_data = next(train_data_iter)

# sample_list, \
# batch_subword_input_ids, batch_attention_mask, batch_token_type_ids, subword2char_span_list, \
# batch_char_input_ids4subword, batch_word_input_ids, batch_subword2word_idx_map, \
# batch_shaking_tag = batch_data

# print(sample_list[0])
# print()
# print(bert_tokenizer.decode(batch_subword_input_ids[0].tolist()))
# print(batch_subword_input_ids.size())
# print(batch_attention_mask.size())
# print(batch_token_type_ids.size())
# print(len(subword2char_span_list))
# print(batch_char_input_ids4subword.size())
# print(batch_word_input_ids.size())
# print(batch_subword2word_idx_map.size())
# print(batch_shaking_tag.size())


# # Model

# In[ ]:


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

if parallel:
    ent_extractor = nn.DataParallel(ent_extractor)
ent_extractor = ent_extractor.to(device)


# # Metrics

# In[ ]:


metrics = Metrics(handshaking_tagger)
metrics4valid = Metrics(handshaking_tagger4valid)


# # Train

# In[ ]:


# train step
def train_step(train_data, optimizer):
    ent_extractor.train()
    
    if bert_config is not None:
        sample_list,         padded_sent_list,         batch_subword_input_ids,         batch_attention_mask,         batch_token_type_ids,         tok2char_span_list,         batch_char_input_ids4subword,         batch_word_input_ids,         batch_subword2word_idx_map,         batch_gold_shaking_tag = train_data
    else:
        sample_list,         padded_sent_list,         batch_char_input_ids4subword,         batch_word_input_ids,         tok2char_span_list,         batch_gold_shaking_tag = train_data
    
    batch_char_input_ids4subword,     batch_word_input_ids,     batch_gold_shaking_tag = (batch_char_input_ids4subword.to(device), 
                              batch_word_input_ids.to(device),
                              batch_gold_shaking_tag.to(device) 
                                 )
    if bert_config is not None:
            batch_subword_input_ids,             batch_attention_mask,             batch_token_type_ids,             batch_subword2word_idx_map = (batch_subword_input_ids.to(device), 
                                          batch_attention_mask.to(device), 
                                          batch_token_type_ids.to(device), 
                                          batch_subword2word_idx_map.to(device))
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
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
    
                
#     set_trace()
    loss = metrics.loss_func(batch_pred_shaking_outputs, batch_gold_shaking_tag)
    
    # bp
#     t1 = time.time()
    loss.backward()
    optimizer.step()
#     print("bp: {}".format(time.time() - t1)) 1s + 
    
    batch_pred_shaking_tag = (batch_pred_shaking_outputs > 0.).long()
    sample_acc = metrics.get_sample_accuracy(batch_pred_shaking_tag, batch_gold_shaking_tag)

    return loss.item(), sample_acc.item()

# valid step
def valid_step(valid_data):
    ent_extractor.eval()

    if bert_config is not None:
        sample_list,         padded_sent_list,         batch_subword_input_ids,         batch_attention_mask,         batch_token_type_ids,         tok2char_span_list,         batch_char_input_ids4subword,         batch_word_input_ids,         batch_subword2word_idx_map,         batch_gold_shaking_tag = valid_data
    else:
        sample_list,         padded_sent_list,         batch_char_input_ids4subword,         batch_word_input_ids,         tok2char_span_list,         batch_gold_shaking_tag = valid_data
    
    batch_char_input_ids4subword,     batch_word_input_ids,     batch_gold_shaking_tag = (batch_char_input_ids4subword.to(device), 
                              batch_word_input_ids.to(device),
                              batch_gold_shaking_tag.to(device) 
                                 )
    if bert_config is not None:
            batch_subword_input_ids,             batch_attention_mask,             batch_token_type_ids,             batch_subword2word_idx_map = (batch_subword_input_ids.to(device), 
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
    
    sample_acc = metrics4valid.get_sample_accuracy(batch_pred_shaking_tag, batch_gold_shaking_tag)
    correct_num, pred_num, gold_num = metrics4valid.get_ent_correct_pred_glod_num(sample_list, tok2char_span_list, 
                                                                                  batch_pred_shaking_tag)
    
    return sample_acc.item(), correct_num, pred_num, gold_num


# In[ ]:


max_f1 = 0.
def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):  
    def train(dataloader, ep):
        t_ep = time.time()
        total_loss, total_sample_acc = 0., 0.
        for batch_ind, train_data in enumerate(dataloader):
            t_batch = time.time()
            
            scheduler.step(ep * len(dataloader) + batch_ind)
            loss, sample_acc = train_step(train_data, optimizer)
            
            total_loss += loss
            total_sample_acc += sample_acc
            
            avg_loss = total_loss / (batch_ind + 1)
            avg_sample_acc = total_sample_acc / (batch_ind + 1)
            
            batch_print_format = "\rexp_name: {}, run_num: {}, epoch: {}/{}, batch: {}/{}, train_loss: {}, t_sample_acc: {}, lr: {}, batch_time: {}------------------------"
            print(batch_print_format.format(experiment_name, run_name, 
                                            ep + 1, num_epoch, 
                                            batch_ind + 1, len(dataloader), 
                                            avg_loss, 
                                            avg_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch), end="")
            
            if config["wandb"] is True and batch_ind % hyper_parameters["log_interval"] == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "train_shaking_tag_acc": avg_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })
                
    def valid(dataloader, ep):
        # valid
        t_ep = time.time()
        total_sample_acc = 0.
        total_correct_num, total_pred_num, total_gold_num = 0., 0., 0.
        for batch_ind, dev_data in enumerate(tqdm(dataloader, desc = "Validating")):
            sample_acc, correct_num, pred_num, gold_num = valid_step(dev_data)
            
            total_sample_acc += sample_acc
            total_correct_num += correct_num
            total_pred_num += pred_num
            total_gold_num += gold_num

        avg_sample_acc = total_sample_acc / len(dataloader)
        precision, recall, f1 = metrics4valid.get_scores(total_correct_num, total_pred_num, total_gold_num)

        log_dict = {
                "val_shaking_tag_acc": avg_sample_acc,
                "val_prec": precision,
                "val_recall": recall,
                "val_f1": f1,
                "time": time.time() - t_ep,
            }
        pprint(log_dict)
        if config["wandb"] is True:
            logger.log(log_dict)
        
        return f1
        
    for ep in range(num_epoch):
        train(train_dataloader, ep)   
        print()
        valid_f1 = valid(dev_dataloader, ep)
        
        global max_f1
        if valid_f1 >= max_f1: 
            max_f1 = valid_f1
            if max_f1 > config["f1_2_save"]: # save the best model
                file_num = len(glob.glob(model_state_dict_dir + "/*.pt"))
                torch.save(ent_extractor.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(file_num)))
        print("Current valid_f1: {}, Best f1: {}".format(valid_f1, max_f1))


# In[ ]:


# optimizer 
optimizer = torch.optim.Adam(ent_extractor.parameters(), lr = init_learning_rate)

if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)
    
elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = decay_steps, gamma = decay_rate)


# In[ ]:


if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))
    
train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, epoch_num)

