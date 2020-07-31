import re
from tqdm import tqdm
import torch
from IPython.core.debugger import set_trace
import copy
import torch
import torch.nn as nn
import json

class BIOSETaggingScheme:
    def __init__(self, types, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        tags = sorted(list("BISE"))
        types = sorted(types)
        self.tag2id = {t:idx for idx, t in enumerate(tags)}
        self.tag2id["O"] = 0
        
        self.id2tag = {idx:t for t, idx in self.tag2id.items()}
        
        self.type2id = {t:idx for idx, t in enumerate(types)}
        self.id2type = {idx:t for t, idx in self.type2id.items()}

    def get_spots(self, sample):
        spots = []
        for ent in sample["entity_list"]:
            type_id = self.type2id[ent["type"]]
            tok_span = ent["tok_span"]
            spots.append((type_id, *tok_span))
        return spots
    
    def spots2tags(self, batch_spots):
        tags = torch.zeros([len(batch_data), len(self.type2id), len(self.max_seq_len)]).long()
        for batch_idx, spots in enumerate(batch_spots):
            for spot in spots:
                tok_span = [spots[1], spots[2]]
                type_id = spots[0]
                if tok_span[1] - tok_span[0] == 1:
                    tags[batch_idx][type_id][tok_span[0]] = self.tag2id["S"]
                else:
                    tags[batch_idx][type_id][tok_span[0]] = self.tag2id["B"]
                    tags[batch_idx][type_id][tok_span[1] - 1] = self.tag2id["E"]
                    for i in range(tok_span[0] + 1, tok_span[1] - 1):
                        tags[batch_idx][type_id][i] = self.tag2id["I"]
        return tags
    
    def decode_ent(self, text, tags, tok2char_span, tok_offset = 0, char_offset = 0):
        '''
        tags: size = (type_num, max_seq_len)
        if text is a subtext of test data, tok_offset and char_offset must be set
        '''
        entities = []
        entity_memory_set = set()
        for tp_idx in range(len(tags)):
            tag_seq = [self.id2tag[t_id] for t_id in tags[tp_idx]]
            tag_seq_str = "".join(tag_seq)
            entity_it = re.finditer("(BI*E)|S", tag_seq_str) # find BIE and S entity
            for m in entity_it:
                tok_span = m.span()
                char_spans = tok2char_span[tok_span[0]:tok_span[1]]
                char_sp = [char_spans[0][0], char_spans[-1][1]]
                ent = text[char_sp[0]:char_sp[1]]
                ent_memory = "{}\u2E80{}\u2E80{}\u2E80{}".format(ent, *tok_span)
                if ent_memory not in entity_memory_set:
                    entities.append({
                        "text": ent,
                        "tok_span": [tok_span[0] + tok_offset, tok_span[1] + tok_offset],
                        "char_span": [char_sp[0] + char_offset, char_sp[1] + char_offset],
                        "type": self.id2type[tp_idx],
                    })
                    entity_memory_set.add(ent_memory)
        return entities

class DataMaker4BERT:
    def __init__(self, tagger, tokenizer):
        super().__init__()
        self.tagger = tagger
        self.tokenizer = tokenizer
        
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for sample in tqdm(data, desc = "Generate indexed data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = False,
                                    max_length = max_seq_len, 
                                    pad_to_max_length = True)


            # get spots
            spots = None
            if data_type != "test":
                spots = self.tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample, 
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_batch(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        batch_spots = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            if data_type != "test":
                batch_spots.append(tp[5])

        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        
        batch_tags = None
        if data_type != "test":
            batch_tags = self.tagger.spots2tags(batch_spots)

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_tags  
    
class DataMaker4BiLSTM:
    def __init__(self, text2indices, get_tok2char_span_map, tagger):
        super().__init__()
        self.text2indices = text2indices
        self.get_tok2char_span_map = get_tok2char_span_map
        self.tagger = tagger
        
    def get_indexed_data(self, data, max_seq_len, data_type = "train"):
        indexed_samples = []
        for sample in tqdm(data, desc = "Generate indexed data"):
            text = sample["text"]

            # get spots
            spots = None
            if data_type != "test":
                spots = self.tagger.get_spots(sample)

            # get codes
            input_ids = self.text2indices(text, max_seq_len)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))

            sample_tp = (sample, 
                     input_ids,
                     tok2char_span,
                     spots,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_batch(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []
        batch_spots = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])      
            tok2char_span_list.append(tp[2])
            if data_type != "test":
                batch_spots.append(tp[3])

        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        
        batch_tags = None
        if data_type != "test":
            batch_tags = self.tagger.spots2tags(batch_spots)

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_tags 

# CRF 还没加
class BilstmCrfBiose(nn.Module):
    def __init__(self, 
             init_word_embedding_matrix, 
             emb_dropout_rate,
             rnn_layers,
             hidden_size,
             rnn_dropout_rate,
             type_num):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze = False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.lstm = nn.LSTM(init_word_embedding_matrix.size()[-1], 
                    hidden_size // 2, 
                    num_layers = rnn_layers, 
                    dropout = rnn_dropout_rate,
                    bidirectional = True, 
                    batch_first = True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        self.tagger_fc_list = [nn.Linear(hidden_size, 5) for _ in range(type_num)]
        
        for ind, fc in enumerate(self.tagger_fc_list):
            self.register_parameter("weight_4_tagger{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tagger{}".format(ind), fc.bias)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, hidden_size)
        lstm_outputs, _ = self.lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)

        outputs_list = []
        for fc in self.tagger_fc_list:
            outputs_list.append(fc(lstm_outputs))
        
        outputs = torch.stack(outputs_list, dim = 1)
        return outputs
    
# metrics 还没改  
class Metrics:
    def __init__(self, tagger):
        super().__init__()
        self.tagger = tagger
    
    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * y_pred # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12 # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1]) # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim = -1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim = -1)
        neg_loss = torch.logsumexp(y_pred_neg, dim = -1) # +1: si - (-1), make -1 > si
        pos_loss = torch.logsumexp(y_pred_pos, dim = -1) # +1: 1 - sj, make sj > 1
        return (neg_loss + pos_loss).mean()
    
    def loss_func(self, y_pred, y_true):
        return self._multilabel_categorical_crossentropy(y_pred, y_true)
    
    def get_sample_accuracy(self, pred, truth):
        '''
        tag全等正确率
        '''
    #     # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
    #     pred_id = torch.argmax(pred, dim = -1).int()

        # (batch_size, ..., seq_len) -> (batch_size, -1)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_, axis=0)

        return sample_acc
    
    def get_ent_correct_pred_glod_num(self,gold_sample_list, 
                              tok2char_span_list, 
                              batch_pred_ent_shaking_seq_tag):           

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(gold_sample_list)):
            gold_sample = gold_sample_list[ind]
            text = gold_sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_seq_tag = batch_pred_ent_shaking_seq_tag[ind]
            pred_entities = self.tagger.decode_ent(text, pred_ent_shaking_seq_tag, tok2char_span)
            gold_entities = gold_sample["entity_list"]

            pred_num += len(pred_entities)
            gold_num += len(gold_entities)

            memory_set = set()
            for ent in gold_entities:
                memory_set.add("{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"]))

            for ent in pred_entities:
                hit = "{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"])
                if hit in memory_set:
                    correct_num += 1

        return correct_num, pred_num, gold_num
    
    def get_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1
    
