import re
from tqdm import tqdm
import torch
from IPython.core.debugger import set_trace
import copy
import torch
import torch.nn as nn
import json
from NER.ner_common.components import HandshakingKernel
from torch.nn.parameter import Parameter
from transformers import AutoModel
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
import numpy as np
import pandas as pd
import word2vec

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
        for point in torch.nonzero(shaking_tag, as_tuple = False): #  shaking_tag.nonzero() -> torch.nonzero(shaking_tag, as_tuple = False)
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
                 text2char_indices_func,
                 max_word_num, 
                 max_subword_num,
                 max_char_num_in_tok 
                 ):
        super().__init__()
        self.handshaking_tagger = handshaking_tagger
        
        self.word_tokenizer = word_tokenizer
        self.subword_tokenizer = subword_tokenizer
        self.text2char_indices_func = text2char_indices_func
        
        self.max_word_num = max_word_num
        self.max_subword_num = max_subword_num
        self.max_char_num_in_tok = max_char_num_in_tok

    def get_indexed_data(self, data, data_type = "train"):
        '''
        indexing data
        if data_type == "test", matrix_spots is not included
        
        '''      
        indexed_sample_list = []
        max_word_num = self.max_word_num
        max_subword_num = self.max_subword_num
        max_char_num_in_tok = self.max_char_num_in_tok
        
        max_word_num4flair = max([len(sample["text"].split(" ")) for sample in data]) # 这里主要考虑让flair脱离max_word_num的依赖，如果有bug，改回max_word_num。以后用到flair都要带上max_word_num。
        for sample in tqdm(data, desc = "Generate indexed data"):
            text = sample["text"]
            indexed_sample = {}
            indexed_sample["sample"] = sample
            if self.subword_tokenizer is not None:
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
                
                indexed_sample["subword_input_ids"] = subword_input_ids
                indexed_sample["attention_mask"] = attention_mask
                indexed_sample["token_type_ids"] = token_type_ids
                indexed_sample["tok2char_span"] = subword2char_span # token is subword level

            # word level tokenizer
            if self.word_tokenizer is not None: # use word enc
                if self.subword_tokenizer is not None: # also use bert
                    indexed_sample["word_input_ids"] = self.word_tokenizer.text2word_indices(text, max_word_num)
                    
                    # subword2word_idx_map: map subword to corresponding word
                    words = self.word_tokenizer.tokenize(text)
                    subword2word_idx_map = []
                    for wd_idx, wd in enumerate(words):
                        for subwd in self.subword_tokenizer.tokenize(wd):
                            if subwd != "[PAD]":
                                subword2word_idx_map.append(wd_idx)
                    if len(subword2word_idx_map) < max_subword_num:
                        subword2word_idx_map.extend([len(words) - 1] * (max_subword_num - len(subword2word_idx_map)))
                    subword2word_idx_map = torch.tensor(subword2word_idx_map).long()
                    indexed_sample["subword2word_idx_map"] = subword2word_idx_map
                
                else: # do not use bert, but use word enc
                    word_codes = self.word_tokenizer.encode_plus(text, max_word_num)
                    word2char_span = word_codes["offset_mapping"]
                    indexed_sample["tok2char_span"] = word2char_span # token is word level
                    indexed_sample["word_input_ids"] = word_codes["input_ids"]
  
            if self.text2char_indices_func is not None: # use char enc
                char_input_ids = self.text2char_indices_func(text)
                char_input_ids_padded = []
                for span in indexed_sample["tok2char_span"]:
                    char_ids = char_input_ids[span[0]:span[1]]

                    if len(char_ids) < max_char_num_in_tok:
                        char_ids.extend([0] * (max_char_num_in_tok - len(char_ids)))
                    else:
                        char_ids = char_ids[:max_char_num_in_tok]
                    char_input_ids_padded.extend(char_ids)
                char_input_ids_padded = torch.tensor(char_input_ids_padded).long()
                indexed_sample["char_input_ids_padded"] = char_input_ids_padded
            
            # prepare padded sentences for flair embeddings
            words = text.split(" ")
            words.extend(["[PAD]"] * (max_word_num4flair - len(words)))
            indexed_sample["padded_sent"] = Sentence(" ".join(words))
            
            # get spots
            if data_type != "test":
                matrix_spots = self.handshaking_tagger.get_spots(sample)
                indexed_sample["matrix_spots"] = matrix_spots
            
            indexed_sample_list.append(indexed_sample)       
        return indexed_sample_list
    
    def generate_batch(self, batch_data, data_type = "train"):
        batch_dict = {}
        if self.subword_tokenizer is not None:
            subword_input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = [] 
            for dt in batch_data:
                subword_input_ids_list.append(dt["subword_input_ids"])
                attention_mask_list.append(dt["attention_mask"])        
                token_type_ids_list.append(dt["token_type_ids"])  
            batch_dict["subword_input_ids"] = torch.stack(subword_input_ids_list, dim = 0)
            batch_dict["attention_mask"] = torch.stack(attention_mask_list, dim = 0)
            batch_dict["token_type_ids"] = torch.stack(token_type_ids_list, dim = 0)

        if self.word_tokenizer is not None:
            word_input_ids_list = [dt["word_input_ids"] for dt in batch_data]
            batch_dict["word_input_ids"] = torch.stack(word_input_ids_list, dim = 0)
            if self.subword_tokenizer is not None:
                subword2word_idx_map_list = [dt["subword2word_idx_map"] for dt in batch_data]
                batch_dict["subword2word_idx"] = torch.stack(subword2word_idx_map_list, dim = 0)
                
        if self.text2char_indices_func is not None:
            char_input_ids_padded_list = [dt["char_input_ids_padded"] for dt in batch_data]
            batch_dict["char_input_ids"] = torch.stack(char_input_ids_padded_list, dim = 0)
        
        # must
        sample_list = []
        tok2char_span_list = []
        padded_sent_list = []
        matrix_spots_batch = []
        for dt in batch_data:
            sample_list.append(dt["sample"])
            tok2char_span_list.append(dt["tok2char_span"])
            padded_sent_list.append(dt["padded_sent"])

            if data_type != "test":
                matrix_spots_batch.append(dt["matrix_spots"])
        batch_dict["sample_list"] = sample_list
        batch_dict["tok2char_span_list"] = tok2char_span_list
        batch_dict["padded_sents"] = padded_sent_list
        
        # shaking tag
        if data_type != "test":
            batch_dict["shaking_tag"] = self.handshaking_tagger.spots2shaking_tag4batch(matrix_spots_batch)

        return batch_dict
    

class TPLinkerNER(nn.Module):
    def __init__(self, 
             char_encoder_config,
             word_encoder_config,
             flair_config,
             handshaking_kernel_config,
             hidden_size,
             activate_enc_fc,
             entity_type_num,
             bert_config):
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
        bert_config = {
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
        combined_hidden_size = 0
        self.char_encoder_config = char_encoder_config
        if char_encoder_config is not None:
            # char encoder
            char_size = char_encoder_config["char_size"]
            char_emb_dim = char_encoder_config["emb_dim"]
            char_emb_dropout = char_encoder_config["emb_dropout"]
            char_bilstm_hidden_size = char_encoder_config["bilstm_hidden_size"]
            char_bilstm_layers = char_encoder_config["bilstm_layers"]
            char_bilstm_dropout = char_encoder_config["bilstm_dropout"]
            max_char_num_in_subword = char_encoder_config["max_char_num_in_tok"]
            self.char_emb = nn.Embedding(char_size, char_emb_dim)
            self.char_emb_dropout = nn.Dropout(p = char_emb_dropout)
            self.char_lstm_l1 = nn.LSTM(char_emb_dim, 
                           char_bilstm_hidden_size[0] // 2, 
                           num_layers = char_bilstm_layers[0], 
                           dropout = char_bilstm_dropout[0],
                           bidirectional = True,
                           batch_first = True)
            self.char_lstm_dropout = nn.Dropout(p = char_bilstm_dropout[1])
            self.char_lstm_l2 = nn.LSTM(char_bilstm_hidden_size[0], 
                           char_bilstm_hidden_size[1] // 2, 
                           num_layers = char_bilstm_layers[1], 
                           dropout = char_bilstm_dropout[2],
                           bidirectional = True,
                           batch_first = True)
            self.char_cnn = nn.Conv1d(char_bilstm_hidden_size[1], char_bilstm_hidden_size[1], max_char_num_in_subword, stride = max_char_num_in_subword)
            combined_hidden_size += char_bilstm_hidden_size[1]
        
        # word encoder
        ## init word embedding
        self.word_encoder_config = word_encoder_config
        if word_encoder_config is not None:
            word2idx = word_encoder_config["word2idx"]
            word_size = len(word2idx)
            word_emb_key = word_encoder_config["emb_key"]
            word_emb_dropout = word_encoder_config["emb_dropout"]
            word_bilstm_hidden_size = word_encoder_config["bilstm_hidden_size"]
            word_bilstm_layers = word_encoder_config["bilstm_layers"]
            word_bilstm_dropout = word_encoder_config["bilstm_dropout"]
            freeze_word_emb = word_encoder_config["freeze_word_emb"]

            print("Loading pretrained word embeddings...")
            if word_emb_key == "glove":
                glove_df = pd.read_csv('../../pretrained_emb/glove/glove.6B.100d.txt', sep=" ", quoting = 3, header = None, index_col = 0)
                pretrained_emb = {key: val.values for key, val in glove_df.T.items()}
                word_emb_dim = len(list(pretrained_emb.values())[0])
            elif word_emb_key == "pubmed":
                pretrained_emb = word2vec.load('../../pretrained_emb/bio_nlp_vec/PubMed-shuffle-win-30.bin')
                word_emb_dim = len(pretrained_emb.vectors[0])
            init_word_embedding_matrix = np.random.normal(-0.5, 0.5, size=(word_size, word_emb_dim))
            hit_count = 0
            for word, idx in tqdm(word2idx.items(), desc = "Init word embedding matrix"):
                if word in pretrained_emb:
                    hit_count += 1
                    init_word_embedding_matrix[idx] = pretrained_emb[word]
            print("pretrained word embedding hit rate: {}".format(hit_count / word_size))
            init_word_embedding_matrix = torch.FloatTensor(init_word_embedding_matrix)

            ## word encoder model
            self.word_emb = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze = freeze_word_emb)
            self.word_emb_dropout = nn.Dropout(p = word_emb_dropout)
            self.word_lstm_l1 = nn.LSTM(word_emb_dim, 
                             word_bilstm_hidden_size[0] // 2, 
                             num_layers = word_bilstm_layers[0],
                             dropout = word_bilstm_dropout[0],
                             bidirectional = True,
                             batch_first = True)
            self.word_lstm_dropout = nn.Dropout(p = word_bilstm_dropout[1])
            self.word_lstm_l2 = nn.LSTM(word_bilstm_hidden_size[0], 
                             word_bilstm_hidden_size[1] // 2, 
                             num_layers = word_bilstm_layers[1],
                             dropout = word_bilstm_dropout[2],
                             bidirectional = True,
                             batch_first = True)
            combined_hidden_size += word_bilstm_hidden_size[1]
        
        
        # bert 
        self.bert_config = bert_config
        if bert_config is not None: 
            bert_path = bert_config["path"]
            bert_finetune = bert_config["finetune"]
            self.use_last_k_layers_bert = bert_config["use_last_k_layers"]
            self.bert = AutoModel.from_pretrained(bert_path)
            if not bert_finetune: # if train without finetuning bert
                for param in self.bert.parameters():
                    param.requires_grad = False       
#             hidden_size = self.bert.config.hidden_size
            combined_hidden_size += self.bert.config.hidden_size
        
        # flair 
        self.flair_config = flair_config
        if flair_config is not None:
            embedding_models = [FlairEmbeddings(emb_id) for emb_id in flair_config["embedding_ids"]]
            self.stacked_flair_embeddings_model = StackedEmbeddings(embedding_models)
            combined_hidden_size += embedding_models[0].embedding_length * len(embedding_models)
        
        # encoding fc
        self.enc_fc = nn.Linear(combined_hidden_size, hidden_size)
        self.activate_enc_fc = activate_enc_fc
        
        # handshaking kernel
        shaking_type = handshaking_kernel_config["shaking_type"]
        context_type = handshaking_kernel_config["context_type"]
        visual_field = handshaking_kernel_config["visual_field"]
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, context_type, visual_field)
        
        # decoding fc
        self.dec_fc = nn.Linear(hidden_size, entity_type_num)
        
    def forward(self, char_input_ids = None, 
                word_input_ids = None, 
                padded_sents = None, 
                subword_input_ids = None, 
                attention_mask = None, 
                token_type_ids = None, 
                subword2word_idx = None):
        
        # features
        features = []
        # char
        if self.char_encoder_config is not None:
            # char_input_ids: (batch_size, seq_len * max_char_num_in_subword)
            # char_input_emb/char_hiddens: (batch_size, seq_len * max_char_num_in_subword, char_emb_dim)
            # char_conv_oudtut: (batch_size, seq_len, char_emb_dim)
            char_input_emb = self.char_emb(char_input_ids)
            char_input_emb = self.char_emb_dropout(char_input_emb)
            char_hiddens, _ = self.char_lstm_l1(char_input_emb)
            char_hiddens, _ = self.char_lstm_l2(self.char_lstm_dropout(char_hiddens))
            char_conv_oudtut = self.char_cnn(char_hiddens.permute(0, 2, 1)).permute(0, 2, 1)
            features.append(char_conv_oudtut)
        
        # word
        if self.word_encoder_config is not None:
            # word_input_ids: (batch_size, seq_len)
            # word_input_emb/word_hiddens: batch_size, seq_len, word_emb_dim)
            word_input_emb = self.word_emb(word_input_ids)
            word_input_emb = self.word_emb_dropout(word_input_emb)
            word_hiddens, _ = self.word_lstm_l1(word_input_emb)
            word_hiddens, _ = self.word_lstm_l2(self.word_lstm_dropout(word_hiddens))
            if self.bert_config is not None:
                # chose and repeat word hiddens, to align with subword num
                word_hiddens = torch.gather(word_hiddens, 1, subword2word_idx[:,:,None].repeat(1, 1, word_hiddens.size()[-1]))
            features.append(word_hiddens)
            
        # flair embeddings 
        if self.flair_config is not None:
            self.stacked_flair_embeddings_model.embed(padded_sents)
            flair_embeddings = torch.stack([torch.stack([tok.embedding for tok in sent]) for sent in padded_sents])
            if self.bert_config is not None:
                # chose and repeat word hiddens, to align with subword num
                flair_embeddings = torch.gather(flair_embeddings, 1, subword2word_idx[:,:,None].repeat(1, 1, flair_embeddings.size()[-1]))
            features.append(flair_embeddings)
     
        if self.bert_config is not None:
            # subword_input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
            context_oudtuts = self.bert(subword_input_ids, attention_mask, token_type_ids)
            # last_hidden_state: (batch_size, seq_len, hidden_size)
            hidden_states = context_oudtuts[2]
            subword_hiddens = torch.mean(torch.stack(list(hidden_states)[-self.use_last_k_layers_bert:], dim = 0), dim = 0)
            features.append(subword_hiddens)
            
        # combine features
        combined_hiddens = self.enc_fc(torch.cat(features, dim = -1))
        if self.activate_enc_fc:
            combined_hiddens = torch.tanh(combined_hiddens)
        
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # shaking_seq_len: max_seq_len * vf - sum(1, vf)
        shaking_hiddens = self.handshaking_kernel(combined_hiddens)
        
        # ent_shaking_oudtuts: (batch_size, shaking_seq_len, entity_type_num)
        ent_shaking_oudtuts = self.dec_fc(shaking_hiddens)

        return ent_shaking_oudtuts
    
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
        y_pred = y_pred.float()
        y_true = y_true.float()

        y_pred = (1 - 2 * y_true) * y_pred # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12 # mask the pred oudtuts of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred oudtuts of neg classes
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
    
