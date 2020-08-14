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
from transformers import AutoModel

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
    def __init__(self, handshaking_tagger, 
                 word_tokenizer,
                 subword_tokenizer,
                 text2char_indices_func
                 ):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
        self.word_tokenizer = word_tokenizer
        self.subword_tokenizer = subword_tokenizer
        self.text2char_indices_func = text2char_indices_func
        
    def get_indexed_data(self, data, max_word_num, max_subword_num, max_char_num_in_tok, data_type = "train"):
        '''
        indexing data
        '''
        indexed_samples = []
        for sample in tqdm(data, desc = "Generate indexed data"):
            text = sample["text"]
            # codes for bert input
            bert_codes = self.subword_tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = False,
                                    max_length = max_subword_num, 
                                    truncation = True,
                                    pad_to_max_length = True)
        
            # get bert codes
            subword_input_ids = torch.tensor(bert_codes["input_ids"]).long()
            attention_mask = torch.tensor(bert_codes["attention_mask"]).long()
            token_type_ids = torch.tensor(bert_codes["token_type_ids"]).long()
            subword2char_span = bert_codes["offset_mapping"]
            
            # word level tokenizer
            word_input_ids = self.word_tokenizer.text2word_indices(text, max_word_num)
#             word_input_ids = word_codes["input_ids"]
#             word2char_span = word_codes["offset_mapping"]
            
            # char input ids
            char_input_ids = self.text2char_indices_func(text)
            def padding_char_input_ids(offset_mapping, max_char_num_in_tok):
                char_input_ids_padded = []
                for span in offset_mapping:
                    char_ids = char_input_ids[span[0]:span[1]]
                    assert len(char_ids) <= max_char_num_in_tok
                    if len(char_ids) < max_char_num_in_tok:
                        char_ids.extend([0] * (max_char_num_in_tok - len(char_ids)))
                    char_input_ids_padded.extend(char_ids)
                return torch.tensor(char_input_ids_padded).long()
            char_input_ids4subword = padding_char_input_ids(subword2char_span, max_char_num_in_tok)
#             char_input_ids4word = padding_char_input_ids(word2char_span, max_char_num_in_word)
            
#             # word2subword_span
#             words = self.word_tokenizer.tokenize(text)
#             word2subword_span = []
#             subword_num = 0
#             for wd in words:
#                 word2subword_span.append([subword_num, subword_num + len(self.subword_tokenizer.tokenize(wd))])
#             if len(word2subword_span) < max_word_num:
#                 word2subword_span.extend([0, 0] * (max_word_num - len(word2subword_span)))
#             if max_word_num != -1:
#                 word2subword_span = word2subword_span[:max_word_num]
                
            # word_input_ids_repeat
            words = self.word_tokenizer.tokenize(text)
            subword2word_idx_map = []
            for wd_idx, wd in enumerate(words):
                for subwd in self.subword_tokenizer.tokenize(wd):
                    if subwd != "[PAD]":
                        subword2word_idx_map.append(wd_idx)
            if len(subword2word_idx_map) < max_subword_num:
                subword2word_idx_map.extend([len(words) - 1] * (max_subword_num - len(subword2word_idx_map)))
            subword2word_idx_map = torch.tensor(subword2word_idx_map).long()
            
            
            # get spots
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.handshaking_tagger.get_spots(sample)
                
            sample_tp = (sample, 
                     subword_input_ids,
                     attention_mask,
                     token_type_ids,
                     subword2char_span,
                     char_input_ids4subword,
                     word_input_ids,
                     subword2word_idx_map,
#                      char_input_ids4word, 
                     matrix_spots
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_batch(self, batch_data, data_type = "train"):
        sample_list = []
        subword_input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        subword2char_span_list = []
        char_input_ids4subword_list = []
        word_input_ids_list = []
        subword2word_idx_map_list = []
        matrix_spots_batch = []

        for tp in batch_data:
            sample_list.append(tp[0])
            subword_input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])  
            subword2char_span_list.append(tp[4])
            char_input_ids4subword_list.append(tp[5])
            word_input_ids_list.append(tp[6])
            subword2word_idx_map_list.append(tp[7])
            if data_type != "test":
                matrix_spots_batch.append(tp[8])

        batch_subword_input_ids = torch.stack(subword_input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        
        batch_char_input_ids4subword = torch.stack(char_input_ids4subword_list, dim = 0)
        batch_word_input_ids = torch.stack(word_input_ids_list, dim = 0)
        batch_subword2word_idx_map = torch.stack(subword2word_idx_map_list, dim = 0)
        
        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(matrix_spots_batch)

        return sample_list, \
            batch_subword_input_ids, \
            batch_attention_mask, \
            batch_token_type_ids, \
            subword2char_span_list, \
            batch_char_input_ids4subword, \
            batch_word_input_ids, \
            batch_subword2word_idx_map, \
            batch_shaking_tag
    

class TPLinkerNER(nn.Module):
    def __init__(self, 
             char_encoder_config,
             bert_cofig,
             word_encoder_config,
             handshaking_kernel_config,
             activate_enc_fc,
             entity_type_num):
        super().__init__()
        '''
        char_encoder_config = {
            "char_size": len(char2idx), # 
            "emb_dim": char_emb_dim,
            "emb_dropout": char_emb_dropout,
            "bilstm_layers": char_bilstm_layers,
            "bilstm_dropout": char_bilstm_dropout,
            "max_char_num_in_tok": max_char_num_in_tok,
        }
        bert_cofig = {
            "path": encoder_path,
            "fintune": bert_finetune,
            "use_last_k_layers": use_last_k_layers_hiddens,
        }
        word_encoder_config = {
            "init_word_embedding_matrix": init_word_embedding_matrix,
            "emb_dropout": word_emb_dropout,
            "bilstm_layers": word_bilstm_layers,
            "bilstm_dropout": word_bilstm_dropout,
            "freeze_word_emb": freeze_word_emb,
        }

        handshaking_kernel_config = {
            "shaking_type": hyper_parameters["shaking_type"],
            "context_type": hyper_parameters["context_type"],
            "visual_field": visual_field, # 
        }
        '''

        # bert 
        bert_path = bert_cofig["path"]
        bert_finetune = bert_cofig["fintune"]
        self.use_last_k_layers_bert = bert_cofig["use_last_k_layers"]
        self.bert = AutoModel.from_pretrained(bert_path)
        if not bert_finetune: # if train without finetuning bert
            for param in self.bert.parameters():
                param.requires_grad = False       
        bert_hidden_size = self.bert.config.hidden_size
        
        # char encoder
        char_size = char_encoder_config["char_size"]
        char_emb_dim = char_encoder_config["emb_dim"]
        char_emb_dropout = char_encoder_config["emb_dropout"]
        char_bilstm_layers = char_encoder_config["bilstm_layers"]
        char_bilstm_dropout = char_encoder_config["bilstm_dropout"]
        max_char_num_in_subword = char_encoder_config["max_char_num_in_tok"]
        self.char_emb = nn.Embedding(char_size, char_emb_dim)
        self.char_emb_dropout = nn.Dropout(p = char_emb_dropout)
        self.char_lstm = nn.LSTM(char_emb_dim, 
                       char_emb_dim // 2, 
                       num_layers = char_bilstm_layers, 
                       dropout = char_bilstm_dropout,
                       bidirectional = True,
                       batch_first = True)
        self.char_cnn = nn.Conv1d(char_emb_dim, char_emb_dim, max_char_num_in_subword, stride = max_char_num_in_subword)
        
        # word encoder
        init_word_embedding_matrix = word_encoder_config["init_word_embedding_matrix"]
        word_emb_dropout = word_encoder_config["emb_dropout"]
        word_bilstm_layers = word_encoder_config["bilstm_layers"]
        word_bilstm_dropout = word_encoder_config["bilstm_dropout"]
        freeze_word_emb = word_encoder_config["freeze_word_emb"]
        self.word_emb = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze = freeze_word_emb)
        self.word_emb_dropout = nn.Dropout(p = word_emb_dropout)
        word_emb_dim = init_word_embedding_matrix.size()[-1]
        self.word_lstm = nn.LSTM(word_emb_dim, 
                         word_emb_dim // 2, 
                         num_layers = word_bilstm_layers,
                         dropout = word_bilstm_dropout,
                         bidirectional = True,
                         batch_first = True)
        
        # encoding fc
        self.enc_fc = nn.Linear(bert_hidden_size + word_emb_dim + char_emb_dim, bert_hidden_size)
        self.activate_enc_fc = activate_enc_fc
        
        # handshaking kernel
        shaking_type = handshaking_kernel_config["shaking_type"]
        context_type = handshaking_kernel_config["context_type"]
        visual_field = handshaking_kernel_config["visual_field"]
        self.handshaking_kernel = HandshakingKernel(bert_hidden_size, shaking_type, context_type, visual_field)
        
        # decoding fc
        self.dec_fc = nn.Linear(bert_hidden_size, entity_type_num)
        
    def forward(self, char_input_ids, subword_input_ids, attention_mask, token_type_ids, word_input_ids, subword2word_idx):
        # subword_input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.bert(subword_input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, bert_hidden_size)
        hidden_states = context_outputs[2]
        subword_hiddens = torch.mean(torch.stack(list(hidden_states)[-self.use_last_k_layers_bert:], dim = 0), dim = 0)
        
        # char_input_ids: (batch_size, seq_len * max_char_num_in_subword)
        # char_input_emb/char_hiddens: (batch_size, seq_len * max_char_num_in_subword, char_emb_dim)
        # char_conv_output: (batch_size, seq_len, char_emb_dim)
        char_input_emb = self.char_emb(char_input_ids)
        char_input_emb = self.char_emb_dropout(char_input_emb)
        char_hiddens, _ = self.char_lstm(char_input_emb)
        char_conv_output = self.char_cnn(char_hiddens.permute(0, 2, 1)).permute(0, 2, 1)
        
        # word_input_ids: (batch_size, seq_len)
        # word_input_emb/word_hiddens: batch_size, seq_len, word_emb_dim)
        word_input_emb = self.word_emb(word_input_ids)
        word_input_emb = self.word_emb_dropout(word_input_emb)
        word_hiddens, _ = self.word_lstm(word_input_emb)
        word_chosen_hiddens = torch.gather(word_hiddens, 1, subword2word_idx[:,:,None].repeat(1, 1, word_hiddens.size()[-1]))
        
        combined_hiddens = self.enc_fc(torch.cat([char_conv_output, subword_hiddens, word_chosen_hiddens], dim = -1))
        if self.activate_enc_fc:
            combined_hiddens = torch.tanh(combined_hiddens)
        
        # shaking_hiddens: (batch_size, shaking_seq_len, bert_hidden_size)
        # shaking_seq_len: max_seq_len * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(combined_hiddens)
        
        # ent_shaking_outputs: (batch_size, shaking_seq_len, entity_type_num)
        ent_shaking_outputs = self.dec_fc(shaking_hiddens)

        return ent_shaking_outputs
    
class Metrics:
    def __init__(self, handshaking_tagger):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
        self.last_weights = None
        
    def GHM(self, gradient, bins = 10, beta = 0.9):
        '''
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        '''
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid((gradient - avg) / std) # normalization and pass through sigmoid to 0 ~ 1.
        
        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(gradient_norm, 0, 0.9999999) # ensure elements in gradient_norm != 1.
        
        example_sum = torch.flatten(gradient_norm).size()[0] # N

        # calculate weights    
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0 # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() + example_sum / bins )
        # EMA: exponential moving averaging
#         print()
#         print("hits_vec: {}".format(hits_vec))
#         print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(gradient.device) # init by ones
        current_weights = self.last_weights * beta + (1 - beta) * current_weights
        self.last_weights = current_weights
#         print("ema current_weights: {}".format(current_weights))
        
        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(gradient_norm.size()[0], gradient_norm.size()[1], 1)
        weights4examples = torch.gather(weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient # return weighted gradients

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
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
        return (self.GHM(neg_loss + pos_loss, bins = 1000)).sum() 
    
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
    
