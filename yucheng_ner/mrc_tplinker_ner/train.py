#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import AutoModel, BertTokenizerFast
from IPython.core.debugger import set_trace
import copy
from tqdm import tqdm
import html
from pprint import pprint
import glob
import time
from ner_common.utils import Preprocessor
from mrc_tplinker_ner import (HandshakingTaggingScheme, 
                          DataMaker, 
                          MRCTPLinkerNER, 
                          Metrics)
import json
import wandb
import yaml


# # Superparameter

# In[ ]:


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
config = yaml.load(open("train_config.yaml", "r"), Loader = yaml.FullLoader)
# hyperparameters
hyper_parameters = config["hyper_parameters"]


# In[ ]:

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# device
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for reproductivity
torch.manual_seed(hyper_parameters["seed"]) # pytorch random seed
torch.backends.cudnn.deterministic = True


# In[ ]:


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


data_home = config["data_home"]

meta_path = os.path.join(data_home, experiment_name, config["meta"])
meta = json.load(open(meta_path, "r", encoding = "utf-8"))

type2questions_path = os.path.join(data_home, experiment_name, config["type2questions"])
type2questions = json.load(open(type2questions_path, "r", encoding = "utf-8"))
tags = meta["tags"]

batch_size = hyper_parameters["batch_size"]
parallel = hyper_parameters["parallel"]
if parallel:
    print("Parallel training is set up!")
epoch_num = hyper_parameters["epochs"]
visual_field = hyper_parameters["visual_field"]

encoder_path = config["bert_path"]
tokenizer = BertTokenizerFast.from_pretrained(encoder_path, add_special_tokens = False, do_lower_case = False)

max_seq_len_t1 = hyper_parameters["max_seq_len"] # max sequence length of text 1
pred_max_seq_len_t1 = hyper_parameters["pred_max_seq_len"]
max_seq_len_question = max([len(tokenizer.tokenize(q)) for questions in type2questions.values() for q in questions])
max_seq_len = max_seq_len_t1 + max_seq_len_question + 1  # +1: [SEP]
pred_max_seq_len = pred_max_seq_len_t1 + max_seq_len_question + 1
assert max_seq_len <= 512 and pred_max_seq_len <= 512
train_sliding_len = hyper_parameters["sliding_len"]
pred_sliding_len = hyper_parameters["pred_sliding_len"]

init_learning_rate = float(hyper_parameters["lr"])

train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])


# # Load Data

# In[ ]:


train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))


# # Split

# In[ ]:


def get_tok2char_span_map(text):
    tok2char_span = tokenizer.encode_plus(text, 
                                           return_offsets_mapping = True, 
                                           add_special_tokens = False)["offset_mapping"]
    return tok2char_span

tokenize = lambda text: tokenizer.tokenize(text)

# preprocessor
preprocessor = Preprocessor(tokenize, get_tok2char_span_map)


# In[ ]:


def split(data, max_seq_len, sliding_len, data_type = "train"):
    '''
    split into short texts
    '''
    max_tok_num = 0
    for sample in tqdm(data, "calculating the max token number of {} data".format(data_type)):
        text = sample["text"]
        tokens = tokenizer.tokenize(text)
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
short_train_data = split(train_data, max_seq_len_t1, train_sliding_len, "train")
short_valid_data = split(valid_data, pred_max_seq_len_t1, pred_sliding_len, "valid")


# In[ ]:


# # check tok spans of new short article dict list
# for art_dict in tqdm(short_train_data + short_valid_data):
#     text = art_dict["text"]
#     tok2char_span = get_tok2char_span_map(text)
#     for term in art_dict["entity_list"]:        
#         # # voc 里必须加两个token：hypo, mineralo
#         tok_span = term["tok_span"]
#         char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
#         pred_text = text[char_span_list[0][0]:char_span_list[-1][1]]
#         assert pred_text == term["text"]


# # Tagging

# In[ ]:


if meta["visual_field_rec"] > visual_field:
    visual_field = meta["visual_field_rec"]
    print("Recommended visual_field is greater than current visual_field, reset to rec val: {}".format(visual_field))


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


handshaking_tagger = HandshakingTaggingScheme(tags, max_seq_len_t1, visual_field)
handshaking_tagger4valid = HandshakingTaggingScheme(tags, pred_max_seq_len_t1, visual_field)


# In[ ]:


# # check tagging and decoding
# def check_tagging_n_decoding(data4check, handshaking_tagger):
#     for sample in tqdm(data4check):
#         batch_matrix_spots = []
#         batch_sub_samples = []

#         type2spots = handshaking_tagger.get_spots(sample)
#         for type_, questions in type2questions.items():
#             for q in questions:
#                 sub_sample = copy.deepcopy(sample)
#                 sub_sample["entity_list"] = [ent for ent in sample["entity_list"] if ent["type"] == type_]
#                 sub_sample["question"] = q
#                 batch_sub_samples.append(sub_sample)
#                 batch_matrix_spots.append(type2spots[type_])

#     #     set_trace()
#         # tagging
#         # batch_shaking_tag: (batch_size, shaking_tag, tag_size)
#         batch_shaking_tag = handshaking_tagger.spots2shaking_tag4batch(batch_matrix_spots)
#     #     %timeit shaking_tagger.spots2shaking_tag4batch(batch_matrix_spots) #0.3s

#         # decoding
#         ent_list = []
#         for batch_idx in range(len(batch_sub_samples)):
#             sub_sample = batch_sub_samples[batch_idx]
#             text = sub_sample["text"]
#             shaking_tag = batch_shaking_tag[batch_idx]
#             # decode
#             tok2char_span = get_tok2char_span_map(text)
#             ent_list.extend(handshaking_tagger.decode_ent(sub_sample["question"], text, shaking_tag, tok2char_span))
#         gold_sample = sample
#         pred_sample = copy.deepcopy(gold_sample)
#         pred_sample["entity_list"] = ent_list

#         if not sample_equal_to(pred_sample, gold_sample):
#             set_trace()
#         if not sample_equal_to(gold_sample, pred_sample):
#             set_trace()
        
# check_tagging_n_decoding(short_train_data, handshaking_tagger)
# check_tagging_n_decoding(short_valid_data, handshaking_tagger4valid)


# # Dataset

# In[ ]:


data_maker = DataMaker(handshaking_tagger, tokenizer)
data_maker4valid = DataMaker(handshaking_tagger4valid, tokenizer)


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


# In[ ]:


indexed_train_sample_list = data_maker.get_indexed_data(short_train_data, max_seq_len, type2questions)
indexed_valid_sample_list = data_maker4valid.get_indexed_data(short_valid_data, pred_max_seq_len, type2questions)


# In[ ]:


train_dataloader = DataLoader(MyDataset(indexed_train_sample_list), 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  num_workers = 6,
                                  drop_last = False,
                                  collate_fn = data_maker.generate_batch,
                                 )
valid_dataloader = DataLoader(MyDataset(indexed_valid_sample_list), 
                          batch_size = batch_size, 
                          shuffle = False, 
                          num_workers = 6,
                          drop_last = False,
                          collate_fn = data_maker4valid.generate_batch,
                         )


# In[ ]:


# # have a look at dataloader
# train_data_iter = iter(train_dataloader)
# batch_data = next(train_data_iter)
# sample_list, batch_input_ids, \
# batch_attention_mask, batch_token_type_ids, \
# tok2char_span_list, batch_shaking_tag = batch_data

# print(sample_list[0])
# print()
# print(tokenizer.decode(batch_input_ids[0].tolist()))
# print(batch_input_ids.size())
# print(batch_attention_mask.size())
# print(batch_token_type_ids.size())
# print(len(tok2char_span_list))
# print(batch_shaking_tag.size())


# # Model

# In[ ]:


encoder = AutoModel.from_pretrained(encoder_path)


# In[ ]:


fake_input = torch.zeros([batch_size, max_seq_len, encoder.config.hidden_size]).to(device)
shaking_type = hyper_parameters["shaking_type"]
pooling_type = hyper_parameters["pooling_type"]
ent_extractor = MRCTPLinkerNER(encoder, 
                               fake_input, 
                               shaking_type,
                               pooling_type,
                               visual_field)
if parallel:
    ent_extractor = nn.DataParallel(ent_extractor)
ent_extractor = ent_extractor.to(device)


# In[ ]:


metrics = Metrics(handshaking_tagger)
metrics4valid = Metrics(handshaking_tagger4valid)

def bias_loss(weights = None):
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device)
    cross_en = nn.CrossEntropyLoss(weight = weights)  
    return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))
loss_func = bias_loss()


# In[ ]:


# train step
def train_step(train_data, optimizer):
    ent_extractor.train()
    
    sample_list,     batch_input_ids, batch_attention_mask, batch_token_type_ids,     offset_map_list,    batch_gold_shaking_tag = train_data
    
    batch_input_ids, batch_attention_mask, batch_token_type_ids,     batch_gold_shaking_tag = (batch_input_ids.to(device), 
                                  batch_attention_mask.to(device), 
                                  batch_token_type_ids.to(device), 
                                  batch_gold_shaking_tag.to(device), 
                                 )
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    batch_pred_shaking_outputs = ent_extractor(batch_input_ids, batch_attention_mask, batch_token_type_ids, max_seq_len_t1)
#     set_trace()
    loss = loss_func(batch_pred_shaking_outputs, batch_gold_shaking_tag)
    
    # bp
#     t1 = time.time()
    loss.backward()
    optimizer.step()
#     print("bp: {}".format(time.time() - t1)) 1s + 
    
    batch_pred_shaking_tag = torch.argmax(batch_pred_shaking_outputs, dim = -1)
    sample_acc = metrics.get_sample_accuracy(batch_pred_shaking_tag, batch_gold_shaking_tag)

    return loss.item(), sample_acc.item()

# valid step
def valid_step(valid_data):
    ent_extractor.eval()
    
    sample_list,     batch_input_ids, batch_attention_mask, batch_token_type_ids,     offset_map_list,    batch_gold_shaking_tag = valid_data
    
    batch_input_ids, batch_attention_mask, batch_token_type_ids,     batch_gold_shaking_tag = (batch_input_ids.to(device), 
                              batch_attention_mask.to(device), 
                              batch_token_type_ids.to(device), 
                              batch_gold_shaking_tag.to(device), 
                             )
    
    with torch.no_grad():
        batch_pred_shaking_outputs = ent_extractor(batch_input_ids, batch_attention_mask, batch_token_type_ids, pred_max_seq_len_t1)
        
    batch_pred_shaking_tag = torch.argmax(batch_pred_shaking_outputs, dim = -1)
    
    sample_acc = metrics4valid.get_sample_accuracy(batch_pred_shaking_tag, batch_gold_shaking_tag)
    correct_num, pred_num, gold_num = metrics4valid.get_ent_correct_pred_glod_num(sample_list, offset_map_list, 
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

