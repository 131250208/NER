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
    def __init__(self, types, max_seq_len_t1, visual_field):
        '''
        max_seq_len_t1: max sequence length of text 1
        visual_field: how many tokens afterwards need to be taken into consider
        '''
        super().__init__()
        self.visual_field = visual_field
        self.types = set(types)
        
        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len_t1
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:ind + visual_field]]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_idx, matrix_idx in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_idx[0]][matrix_idx[1]] = shaking_idx

    def get_spots(self, sample):
        '''
        type2spots: a dict mapping type to spots
        spot: (start_pos, end_pos - 1)， points in the shaking matrix
        '''
        type2spots = {t: [] for t in self.types}
        for ent in sample["entity_list"]:
            t = ent["type"]
            spot = (ent["tok_span"][0], ent["tok_span"][1] - 1)
            type2spots[t].append(spot) # term["tok_span"][1] - 1: span[1] is not included

        return type2spots
    
    def spots2shaking_tag4batch(self, spots_batch):
        '''
        convert spots to shaking tag
        spots_batch:
            [spots1, spots2, ....]
            spots: [(start_pos, end_pos), ]
        return: shaking tag
        '''
        shaking_seq_len = self.matrix_size * self.visual_field - self.visual_field * (self.visual_field - 1) // 2
#         set_trace()
        shaking_seq_tag = torch.zeros([len(spots_batch), shaking_seq_len]).long()
        for batch_idx, spots in enumerate(spots_batch):
            for sp in spots:
                shaking_ind = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                shaking_seq_tag[batch_idx][shaking_ind] = 1
        return shaking_seq_tag
    
    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag: (shaking_seq_len, )
        return matrix_spots: [(start_pos, end_pos), ]
        '''
        matrix_spots = []
        for point in shaking_tag.nonzero():
            shaking_idx = point[0].item()
            matrix_points = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (matrix_points[0], matrix_points[1])
            matrix_spots.append(spot)
        return matrix_spots
    
    def decode_ent(self, question, text, shaking_tag, tok2char_span, tok_offset = 0, char_offset = 0):
        '''
        shaking_tag: size = (shaking_seq_len, tag_size)
        if text is a subtext of test data, tok_offset and char_offset must be set
        '''
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)
        entities = []
        entity_memory_set = set()
        type_ = re.match("Find (.*?) in the text.*", question).group(1)
        for sp in matrix_spots:
            char_spans = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_spans[0][0], char_spans[-1][1]]
            ent = text[char_sp[0]:char_sp[1]]
            ent_memory = "{}\u2E80{}\u2E80{}".format(ent, *sp)
            if ent_memory not in entity_memory_set:
                entities.append({
                    "text": ent,
                    "tok_span": [sp[0] + tok_offset, sp[1] + 1 + tok_offset],
                    "char_span": [char_sp[0] + char_offset, char_sp[1] + char_offset],
                    "type": type_,
                })
                entity_memory_set.add(ent_memory)
        return entities

class DataMaker:
    def __init__(self, handshaking_tagger, tokenizer):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
        self.tokenizer = tokenizer
        
    def get_indexed_data(self, data, max_seq_len, type2questions, data_type = "train"):
        indexed_samples = []
        for sample in tqdm(data, desc = "Generate indexed data"):
            text = sample["text"]

            # get spots
            type2spots = None
            if data_type != "test":
                type2spots = self.handshaking_tagger.get_spots(sample)
            
            for type_, questions in type2questions.items():
                for question in questions:
                    # codes for bert input
                    text_n_question = "{}[SEP]{}".format(text, question)
                    codes = self.tokenizer.encode_plus(text_n_question, 
                                            return_offsets_mapping = True, 
                                            add_special_tokens = False,
                                            max_length = max_seq_len, 
                                            pad_to_max_length = True)


                    # get codes
                    input_ids = torch.tensor(codes["input_ids"]).long()
                    attention_mask = torch.tensor(codes["attention_mask"]).long()
                    token_type_ids = torch.tensor(codes["token_type_ids"]).long()
                    offset_map = codes["offset_mapping"]

                    # spots
                    matrix_spots = type2spots[type_]
                    
                    new_sample = copy.deepcopy(sample)
                    new_sample["entity_list"] = [ent for ent in sample["entity_list"] if ent["type"] == type_]
                    new_sample["question"] = question
                    
                    sample_tp = (new_sample,
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
    

class MRCTPLinkerNER(nn.Module):
    def __init__(self, 
             encoder,
             fake_input, 
             shaking_type, 
             visual_field):
        super().__init__()
        self.encoder = encoder
        shaking_hidden_size = encoder.config.hidden_size
        
        self.fc = nn.Linear(shaking_hidden_size, 2)
        
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(visual_field, fake_input, shaking_type)
        
    def forward(self, input_ids, attention_mask, token_type_ids, max_seq_len_t1):
        '''
        max_seq_len_t1: max sequence lenght of text 1
        '''
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
#         set_trace()
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # shaking_seq_len: max_seq_len_t1 * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state[:, :max_seq_len_t1, :]) # only consider text 1, let alone the question
        
        # ent_shaking_outputs: (batch_size, shaking_seq_len, entity_type_num)
        ent_shaking_outputs = self.fc(shaking_hiddens)

        return ent_shaking_outputs
    
class Metrics:
    def __init__(self, handshaking_tagger):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
    
    def get_sample_accuracy(self, pred, truth):
        '''
        tag全等正确率
        '''
        # (batch_size, ..., seq_len) -> (batch_size, -1)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_, axis=0)

        return sample_acc
    
    def get_ent_correct_pred_glod_num(self, gold_sample_list,
                               offset_map_list, 
                               batch_pred_ent_shaking_seq_tag):           

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(gold_sample_list)):
            gold_sample = gold_sample_list[ind]
            question = gold_sample["question"]
            text = gold_sample["text"]
            offset_map = offset_map_list[ind]
            pred_ent_shaking_seq_tag = batch_pred_ent_shaking_seq_tag[ind]
            pred_entities = self.handshaking_tagger.decode_ent(question, text, pred_ent_shaking_seq_tag, offset_map)
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
    
