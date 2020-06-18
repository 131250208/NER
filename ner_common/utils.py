import re
from tqdm import tqdm
from IPython.core.debugger import set_trace
import copy

class Preprocessor:
    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._get_tok2char_span_map = get_tok2char_span_map_func
        self._tokenize = tokenize_func
    
    def clean_data_wo_span(self, ori_data, data_type = "train"):
        '''
        rm duplicate whitespaces
        and add whitespaces around tokens to keep special characters from them
        '''
        def clean_text(text):
            text = re.sub("\s+", " ", text).strip()
            text = re.sub("([A-Za-z0-9]+)", r" \1 ", text)
    #         text = re.sub("(\d+)", r" \1 ", text)
            text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(ori_data, desc = "clean"):
            sample["text"] = clean_text(sample["text"])
            if data_type == "test":
                continue
            for ent in sample["entity_list"]:
                ent["text"] = clean_text(ent["text"])
        return ori_data

    def clean_data_w_span(self, ori_data):
        '''
        add a stake to bad samples and remove them from the clean data
        '''
        bad_samples, clean_data = [], []
        for sample in tqdm(ori_data, desc = "cleaning"):
            text = sample["text"]

            bad = False
            for ent in sample["entity_list"]:
                # rm whitespaces
                p = 0
                while ent["text"][p] == " ":
                    ent["char_span"][0] += 1
                    p += 1
                p = len(ent["text"]) - 1
                while ent["text"][p] == " ":
                    ent["char_span"][1] -= 1
                    p -= 1

                char_span = ent["char_span"]
                if ent["text"] not in text or ent["text"] != text[char_span[0]:char_span[1]]:
                    ent["stake"] = 0
                    bad = True

            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_ent_list = [ent for ent in sample["entity_list"] if "stake" not in ent]
            if len(new_ent_list) > 0:
                sample["entity_list"] = new_ent_list
                clean_data.append(sample)
        return clean_data, bad_samples

    def _get_char2tok_span(self, text):
        tok2char_span = self._get_tok2char_span_map(text)
        # get the number of characters
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        # build a map: char index to token level span
        char2tok_span = [[-1, -1] for _ in range(char_num)] # 除了空格，其他字符均有对应token
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1
        return char2tok_span
    
    def _get_ent2char_spans(self, text, entities):
        entities = sorted(entities, key = lambda x: len(x), reverse = True)
        text_cp = " {} ".format(text[:])
        ent2char_spans = {}
        for ent in entities:
            spans = []
            for m in re.finditer(re.escape(" {} ".format(ent)), text_cp):
                span = [m.span()[0], m.span()[1] - 2]
                spans.append(span)
            ent2char_spans[ent] = spans
        return ent2char_spans
    
    def add_char_span(self, dataset):
        for sample in tqdm(dataset, desc = "Adding char level spans"):
            entities = [ent["text"] for ent in sample["entity_list"]]
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities)
            
            # filter 
            ent_memory_set = set()
            uni_entity_list = []
            for ent in sample["entity_list"]:
                ent_memory = "{}-{}".format(ent["text"], ent["type"])
                if ent_memory not in ent_memory_set:
                    uni_entity_list.append(ent)
                    ent_memory_set.add(ent_memory)
                    
            new_ent_list = []
            for ent in uni_entity_list:
                ent_spans = ent2char_spans[ent["text"]]
                for sp in ent_spans:
                    new_ent_list.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "char_span": sp,
                    })
            sample["entity_list"] = new_ent_list
        
    def add_tok_span(self, data):
        '''
        data: must has char span
        '''
        for sample in tqdm(data, desc = "Adding token level span"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                tok_span_list = char2tok_span[char_span[0]:char_span[1]]
                tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
                ent["tok_span"] = tok_span
                
    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len = 50, encoder = "BERT", data_type = "train"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc = "Splitting"):
            medline_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self._get_tok2char_span_map(text)

            # sliding on token level
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT": # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                tok_spans = tok2char_span[start_ind:end_ind]
                char_span = (tok_spans[0][0], tok_spans[-1][-1])
                sub_text = text[char_span[0]:char_span[1]]

                if data_type == "test":
                    if len(sub_text) > 0:
                        new_sample = {
                            "id": medline_id,
                            "text": sub_text,
                            "tok_offset": start_ind,
                            "char_offset": char_span[0],
                        }
                        new_sample_list.append(new_sample)
                else:
                    sub_entity_list = []
                    for term in sample["entity_list"]:
                        tok_span = term["tok_span"]
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                            new_term = copy.deepcopy(term)
                            new_term["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]
                            new_term["char_span"][0] -= char_span[0]
                            new_term["char_span"][1] -= char_span[0]
                            sub_entity_list.append(new_term)

                    if len(sub_entity_list) > 0:
                        new_sample = {
                            "id": medline_id,
                            "text": sub_text,
                            "entity_list": sub_entity_list,
                        }
                        new_sample_list.append(new_sample)

                if end_ind > len(tokens):
                    break
        return new_sample_list