#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
import os
config = yaml.load(open("extractor_config.yaml", "r"), Loader = yaml.FullLoader)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
import json
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import re
from transformers import BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from NER.ner_common.utils import Preprocessor
from NER.tplinker_ner.tplinker_ner import (HandshakingTaggingScheme,
                                              DataMaker, 
                                              TPLinkerNER)
from collections import OrderedDict
from Doraemon import whois, domain2ip
from IPython.core.debugger import set_trace
from bs4 import BeautifulSoup
import socket


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# >>>>>>>>>data path>>>>>>>>>>>>>>
data_home = config["data_home"]
in_file_name = config["in_file_name"]
out_file_name = config["out_file_name"]

# >>>>>>>>predication config>>>>>>
max_seq_len = config["max_seq_len"]
sliding_len = config["sliding_len"]
batch_size = config["batch_size"]

# >>>>>>>>>>model config>>>>>>>>>>>>
## bert encoder
bert_config = config["bert_config"] if config["use_bert"] else None
bert_config["path"] = data_home + "/bert-base-cased"
use_bert = config["use_bert"]

## handshaking_kernel
handshaking_kernel_config = config["handshaking_kernel_config"]
visual_field = handshaking_kernel_config["visual_field"]

## encoding fc
enc_hidden_size = config["enc_hidden_size"]
activate_enc_fc = config["activate_enc_fc"]


# # Preprocess

# In[ ]:


state_names_abbr = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
                    "KY",
                    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
                    "ND",
                    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
                    ]

state_names = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
               "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
               "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
               "Missouri",
               "Montana", "Nebraska", "Nevada", "New hampshire", "New Jersey", "New Mexico", "New York",
               "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
               "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
               "West Virginia", "Wisconsin", "Wyoming",
               ]
state_abbr_2_full_name = dict(zip(state_names_abbr, state_names))


# In[ ]:


def drop_html(html):
    text = BeautifulSoup(html).get_text(separator=" ")
    return re.sub("\s+", " ", text)


# In[ ]:


def abbr2fullname(text):
    # replace state abbrivation with full name 
    sub_dict = {}
    for m in re.finditer("([A-Z]{2}) (\d{5})", text):
        state_abbr, zipcode = m.group(1), m.group(2)
        if state_abbr not in state_abbr_2_full_name:
            continue
        sub_dict["{} {}".format(state_abbr, zipcode)] = "{} {}".format(state_abbr_2_full_name[state_abbr], zipcode)

    for ori_str, sub_str in sub_dict.items():
        text = re.sub(re.escape(ori_str), sub_str, text)
    return text 

def preprocess(text):
    text = text.strip()
    text = re.sub("©|(&copy;)|copyright|Copyright|（c）|\(c\)", " copyright ", text)# 替换所有的copyright为统一字符串
    # 用空格将单词、数字、其他字符切开，do not split decimal point 
    text = re.sub("([^A-Za-z0-9\.])", r" \1 ", text)
    text = re.sub("(\D)\.(.)", r"\1 . \2", text)
    text = re.sub("(.)\.(\D)", r"\1 . \2", text)
    # 切开错误连接的单词
    text = re.sub("([A-Z]{1}[a-z]+)([A-Z]{1}[a-z]+)", r"\1 \2", text)
    # remove redundant blanks
    text = re.sub("\s+", " ", text).strip() 
    
    text = abbr2fullname(text)
    return text


# In[ ]:


# bert tokenizer
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_config["path"], add_special_tokens = False, do_lower_case = False)
preprocessor = Preprocessor(bert_tokenizer, True)
# split function
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
        print("max token number of {} is greater than the setting, need to split!".format(data_name, data_name, max_seq_len))
        short_data = preprocessor.split_into_short_samples(data, 
                                          max_seq_len, 
                                          sliding_len = sliding_len, 
                                          data_type = "test")
    else:
        short_data = data
        max_seq_len = max_tok_num
        print("max token number of {} is less than the setting, no need to split!".format(data_name, data_name, max_tok_num))
    return short_data, max_seq_len


# In[ ]:


# load data
page_data_path = os.path.join(data_home, in_file_name)
pages = json.load(open(page_data_path, "r", encoding = "utf-8"))
# for page in pages:
#     page["text"] = preprocess(page["text"])

# domain to ip
url_list = [p["url"] for p in pages if "ip" not in p]
domain_name2ip = {}
domain2ip.gethostbyname_fast(url_list, domain_name2ip)


# In[ ]:


# normalize
for idx, page in tqdm(enumerate(pages), desc = "normalizing"):
    page["id"] = idx
    assert "url" in page and ("html" in page or "text" in page)

    if "ip" not in page:
        dn = domain2ip.urlfillter(page["url"])
        page["ip"] = domain_name2ip[dn]

    if "text" not in page:
        page["text"] = preprocess(drop_html(page["html"]))
    else:
        page["text"] = preprocess(page["text"])


# In[ ]:


# split
ori_pages = copy.deepcopy(pages)
short_pages_, max_seq_len = split(ori_pages, max_seq_len, sliding_len, "pages.json")
print("final max_seq_len is {}".format(max_seq_len))

# filter
key_words = ["all right reserved", "&copy;", "©", "copyright"]
short_pages = []
for page_ in tqdm(short_pages_, desc = "filtering pages wo keywords"):
    page = copy.deepcopy(page_)
    if re.search("|".join(key_words), page["text"], flags = re.I):
        short_pages.append(page)
    elif re.search("({})".format("|".join(state_names)) + "\s+\d{5}", page["text"]):
        short_pages.append(page)

print("Valid page number: {}".format(len(short_pages)))


# # Decoder(Tagger)

# In[ ]:


tags = ["city", "detail", "organization", "state", "zipcode"]
handshaking_tagger = HandshakingTaggingScheme(tags, max_seq_len, visual_field)


# # Dataset

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


# In[ ]:


max_subword_num = cal_max_tok_num(short_pages, bert_tokenizer)
print("max_subword_num: {}".format(max_subword_num))


# In[ ]:


data_maker = DataMaker(handshaking_tagger, None, bert_tokenizer, None, None, max_subword_num, None)


# # Model

# In[ ]:


ent_extractor = TPLinkerNER(None,
                            None,
                            None,
                            handshaking_kernel_config,
                            enc_hidden_size,
                            activate_enc_fc,
                            len(tags),
                            bert_config,
                            )
ent_extractor = ent_extractor.to(device)


# # Prediction

# In[ ]:


def filter_duplicates(ent_list):
    ent_memory_set = set()
    filtered_ent_list = []
    for ent in ent_list:
        ent_memory = "{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"])
        if ent_memory not in ent_memory_set:
            filtered_ent_list.append(ent)
            ent_memory_set.add(ent_memory)
    return filtered_ent_list


# In[ ]:


def predict(test_dataloader, ori_test_data):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc = "Predicting"):
        sample_list = batch_test_data["sample_list"]
        tok2char_span_list = batch_test_data["tok2char_span_list"]
        del batch_test_data["sample_list"]
        del batch_test_data["tok2char_span_list"]

        for k, v in batch_test_data.items():
            if k not in {"padded_sents"}:
                batch_test_data[k] = v.to(device)
        with torch.no_grad():
            batch_pred_shaking_outputs = ent_extractor(**batch_test_data)
        batch_pred_shaking_tag = (batch_pred_shaking_outputs > 0.).long()

        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            text_id = sample["id"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]
            tok_offset, char_offset = 0, 0
            tok_offset, char_offset = (sample["tok_offset"], sample["char_offset"]) if "char_offset" in sample else (0, 0)
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
    page_id2ip= {sample["id"]:sample["ip"] for sample in ori_test_data}
    
    merged_pred_sample_list = []
    for text_id, ent_list in text_id2ent_list.items():
        merged_pred_sample_list.append({
            "id": text_id,
            "ip": page_id2ip[text_id],
            "text": text_id2text[text_id],
            "entity_list": filter_duplicates(ent_list),
        })
        
    return merged_pred_sample_list


# In[ ]:


# predict
indexed_test_data = data_maker.get_indexed_data(short_pages, data_type = "test")
test_dataloader = DataLoader(MyDataset(indexed_test_data), 
                          batch_size = batch_size, 
                          shuffle = False, 
                          num_workers = 6,
                          drop_last = False,
                          collate_fn = lambda data_batch: data_maker.generate_batch(data_batch, data_type = "test"),
                         )
# load model
model_path = os.path.join(data_home, config["model_dict"])
model_state_dict = torch.load(model_path)

# if used paralell train, need to rm prefix "module."
new_model_state_dict = OrderedDict()
for key, v in model_state_dict.items():
    key = re.sub("module\.", "", key)
    new_model_state_dict[key] = v
ent_extractor.load_state_dict(new_model_state_dict)
ent_extractor.eval()

# predict
pred_sample_list = predict(test_dataloader, ori_pages)

page_num_w_ents = len([s for s in pred_sample_list if len(s["entity_list"]) > 0])
print("pages_with_entities/total_pages: {}/{}".format(page_num_w_ents, len(ori_pages)))


# # Dict-based Extraction

# In[ ]:


class NERbyDict(object):
    def __init__(self, prefix_2_orgs_dict):
        self.dict = prefix_2_orgs_dict

    def tokenize(self, text):
        text = re.sub("([A-Za-z0-9\-\']+)", r" \1 ", text)
        text = re.sub("\s+", " ", text)
        return text.split(" ")
    
    def ner(self, target_str):
        tokens = self.tokenize(target_str)
        org_name2score = {}
        copyright_inds = []
        
        for idx, t in enumerate(tokens):
            if re.search("&copy;|©|copyright", t, flags = re.I):
                copyright_inds.append([idx, idx + 1])
            if " ".join(tokens[idx: idx + 3]).lower() == "all right reserved": 
                copyright_inds.append([idx, idx + 3])
                
        for ind, s in enumerate(tokens):
            if s in self.dict: # if a token exists in dict(prefix)
                org_names_2_len = self.dict[s]
                
                for length in sorted(set(org_names_2_len.values()), reverse=True): # sorted org name length in descending order
                    if ind + length > len(tokens) or length == 1: # skip excessive length; org name with len == 1 in the dict is unreliable
                        continue
                    start, end = ind, ind + length
                    substr = " ".join(tokens[start: end])
                    if substr in org_names_2_len: # hit
                        score = len(tokens[start: end]) + len(substr) * 0.1
                        if copyright_inds is not None:
                            for cpr_pos in copyright_inds:
                                dis = max(cpr_pos[0] - end, start - cpr_pos[1])
                                score += 100 / (dis + 1)    
                        org_name2score[substr] = max(org_name2score.get(substr, 0), score)

        return sorted(org_name2score.items(), key = lambda x: x[1], reverse=True)


# In[ ]:


print("Loading org name dicts...")
with open(os.path.join(data_home, "org_name_dict_uni.json"), "r", encoding="utf-8") as dict_uni,     open(os.path.join(data_home, "org_name_dict_sb.json"), "r", encoding="utf-8") as dict_sb:
    ner_dict = {**json.load(dict_uni), **json.load(dict_sb)}
print("Org name dictionary is loaded!")


# In[ ]:


org_dict_extractor = NERbyDict(ner_dict)


# In[ ]:


def owner_org_extract(text):
    orgs = org_dict_extractor.ner(text)
    owner = orgs[0][0] if len(orgs) > 0 else None
    return owner


# In[ ]:


id2org_name_by_dict = {}
for sample in tqdm(pred_sample_list, "extracting org name by dict"):
    org = owner_org_extract(sample["text"])
    if org is not None:
        id2org_name_by_dict[sample["id"]] = org


# In[ ]:


ip2org_webpage = {}
for sample in pred_sample_list:
    text = sample["text"]
    
    org2span = {}
    for ent in sample["entity_list"]:
        if ent["type"] == "organization":
            org2span[ent["text"]] = ent["char_span"]

    # choose an org
    cpr_spans = [m.span() for m in re.finditer("&copy;|©|copyright|all right reserved", text, flags = re.I)]
    org2score = {org:0 for org, span in org2span.items()}
    if len(cpr_spans) > 0:
        for org, span in org2span.items():
            score = 0
            for cpr_pos in cpr_spans:
                dis = max(cpr_pos[0] - span[1], span[0] - cpr_pos[1])
                score += 100 / (dis + 1)  
            org2score[org] += score
    orgs = list(sorted(org2score.items(), key = lambda x: x[1], reverse = True))
    if len(orgs) > 0:
        org = orgs[0][0]
    else:
        org = id2org_name_by_dict[sample["id"]] if sample["id"] in id2org_name_by_dict else None
        
    ip2org_webpage[sample["ip"]] = org


# In[ ]:


ip_list = list(domain_name2ip.values())
ip_info_list = whois.extract_org_names_friendly(ip_list)

ip2org_whois = {}
for ip in ip_info_list:
    if ip is not None:
        ip2org_whois[ip["ip"]] = ip["org_name"]


# In[ ]:


ip2loc_clues = {}
for sample in pred_sample_list:
    ip = sample["ip"]
    ip2loc_clues[ip] = []
    for ent in sample["entity_list"]:
        ip2loc_clues[ip].append({
            "type": ent["type"],
            "text": ent["text"],
        })


# In[ ]:


def unique(ent_list):
    new_ent_list = []
    memory = set()
    for ent in ent_list:
        if str(ent) not in memory:
            memory.add(str(ent))
            new_ent_list.append(ent)
    return new_ent_list

ip2loc_clues_final = {}
for ip, org_page in ip2org_webpage.items():
    if ip in ip2org_whois:
        org_whois = ip2org_whois[ip]
        if org_page in  org_whois:
            ip2loc_clues_final[ip] = {
                "org": org_whois,
                "clues_on_webpage": unique(ip2loc_clues[ip]),
            }


# # Save Results

# In[ ]:


save_path = os.path.join(data_home, out_file_name)
with open(save_path, "w", encoding = "utf-8") as file_out:
    json.dump(ip2loc_clues_final, file_out, ensure_ascii = False)

