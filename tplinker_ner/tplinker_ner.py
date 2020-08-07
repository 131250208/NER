import re
from tqdm import tqdm
import torch
from IPython.core.debugger import set_trace
import copy
import torch
import torch.nn as nn
import json
from ner_common.components import HandshakingKernel
from torch.nn.parameter import Parameter

class HandshakingTaggingScheme:
    def __init__(self, tags, max_seq_len, visual_field):
        super().__init__()
        self.visual_field = visual_field
        self.tag2id = {t:idx for idx, t in enumerate(sorted(tags))}
        self.id2tag = {idx:t for t, idx in self.tag2id.items()}
        
        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:ind + visual_field]]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_idx, matrix_idx in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_idx[0]][matrix_idx[1]] = shaking_idx

    def get_spots(self, sample):
        '''
        spot: (start_pos, end_pos, tag_id)
        '''
        # term["tok_span"][1] - 1: span[1] is not included
        spots = [(ent["tok_span"][0], ent["tok_span"][1] - 1, self.tag2id[ent["type"]]) for ent in sample["entity_list"]]
        return spots
    
    def spots2shaking_tag4batch(self, spots_batch):
        '''
        convert spots to shaking tag
        spots_batch:
            [spots1, spots2, ....]
            spots: [(start_pos, end_pos, tag_id), ]
        return: shaking tag
        '''
        shaking_seq_len = self.matrix_size * self.visual_field - self.visual_field * (self.visual_field - 1) // 2
#         set_trace()
        shaking_seq_tag = torch.zeros([len(spots_batch), shaking_seq_len, len(self.tag2id)]).long()
        for batch_idx, spots in enumerate(spots_batch):
            for sp in spots:
                shaking_ind = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                shaking_seq_tag[batch_idx][shaking_ind][sp[2]] = 1
        return shaking_seq_tag
    
    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        return matrix_spots: [(start_pos, end_pos, tag_id), ]
        '''
        matrix_spots = []
        for point in shaking_tag.nonzero():
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            matrix_points = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (matrix_points[0], matrix_points[1], tag_idx)
            matrix_spots.append(spot)
        return matrix_spots
    
    def decode_ent(self, text, shaking_tag, tok2char_span, tok_offset = 0, char_offset = 0):
        '''
        shaking_tag: size = (shaking_seq_len, tag_size)
        if text is a subtext of test data, tok_offset and char_offset must be set
        '''
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)
        entities = []
        entity_memory_set = set()
        for sp in matrix_spots:
            char_spans = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_spans[0][0], char_spans[-1][1]]
            ent = text[char_sp[0]:char_sp[1]]
            tag_id = sp[2]
            ent_memory = "{}\u2E80{}\u2E80{}\u2E80{}".format(ent, *sp)
            if ent_memory not in entity_memory_set:
                entities.append({
                    "text": ent,
                    "tok_span": [sp[0] + tok_offset, sp[1] + 1 + tok_offset],
                    "char_span": [char_sp[0] + char_offset, char_sp[1] + char_offset],
                    "type": self.id2tag[tag_id],
                })
                entity_memory_set.add(ent_memory)
        return entities

class DataMaker:
    def __init__(self, handshaking_tagger, tokenizer):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
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
                                    truncation = True,
                                    pad_to_max_length = True)


            # get spots
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            offset_map = codes["offset_mapping"]

            sample_tp = (sample, 
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     offset_map,
                     matrix_spots,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_batch(self, batch_data, data_type = "train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        offset_map_list = []
        matrix_spots_batch = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            offset_map_list.append(tp[4])
            if data_type != "test":
                matrix_spots_batch.append(tp[5])

        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        
        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(matrix_spots_batch)

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, offset_map_list, batch_shaking_tag  
    

class TPLinkerNER(nn.Module):
    def __init__(self, 
             encoder,
             use_last_k_layers_hiddens,
             add_bilstm_on_the_top,
             bilstm_layers,
             bilstm_dropout,
             entity_type_num, 
             fake_input, 
             shaking_type,
             pooling_type,
             visual_field):
        super().__init__()
        self.encoder = encoder
        self.use_last_k_layers_hiddens = use_last_k_layers_hiddens
        shaking_hidden_size = encoder.config.hidden_size
        self.add_bilstm_on_the_top = add_bilstm_on_the_top
        if add_bilstm_on_the_top:
            self.bilstm = nn.LSTM(shaking_hidden_size, 
                           shaking_hidden_size // 2, 
                           num_layers = bilstm_layers, 
                           dropout = bilstm_dropout,
                           bidirectional = True, 
                           batch_first = True)
        self.fc = nn.Linear(shaking_hidden_size, entity_type_num)
        
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(visual_field, fake_input, shaking_type, pooling_type)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        hidden_states = context_outputs[2]
        last_hidden_state = torch.mean(torch.stack(list(hidden_states)[-self.use_last_k_layers_hiddens:], dim = 0), dim = 0)
        if self.add_bilstm_on_the_top:
            last_hidden_state, _ = self.bilstm(last_hidden_state)
        
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # shaking_seq_len: max_seq_len * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        
        # ent_shaking_outputs: (batch_size, shaking_seq_len, entity_type_num)
        ent_shaking_outputs = self.fc(shaking_hiddens)

        return ent_shaking_outputs
    
class Metrics:
    def __init__(self, handshaking_tagger):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
    
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
        neg_loss = torch.logsumexp(y_pred_neg, dim = -1) 
        pos_loss = torch.logsumexp(y_pred_pos, dim = -1) 
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
                              offset_map_list, 
                              batch_pred_ent_shaking_seq_tag):           

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(gold_sample_list)):
            gold_sample = gold_sample_list[ind]
            text = gold_sample["text"]
            offset_map = offset_map_list[ind]
            pred_ent_shaking_seq_tag = batch_pred_ent_shaking_seq_tag[ind]
            pred_entities = self.handshaking_tagger.decode_ent(text, pred_ent_shaking_seq_tag, offset_map)
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
    
