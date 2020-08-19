from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import time

class CLNGRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.reset_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        self.update_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        self.input_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        
    def forward(self, sequence):
        # sequence: (batch_size, sequence_len, hidden_size)
        hidden = torch.zeros_like(sequence[:, 0, :])
        hiddens = []
        for i in range(sequence.size()[1]):
            current_input = sequence[:, i, :]
            reset_gate = torch.sigmoid(self.reset_cln(current_input, hidden))
            update_gate = torch.sigmoid(self.update_cln(current_input, hidden))
            hidden_candidate = torch.tanh(self.input_cln(current_input, reset_gate * hidden))
            hidden = hidden_candidate * update_gate + (1 - update_gate) * hidden
            hiddens.append(hidden)
        return torch.stack(hiddens, dim = 1), hidden

class CLNRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        
    def forward(self, sequence):
        # sequence: (batch_size, sequence_len, hidden_size)
        hidden = torch.zeros_like(sequence[:, 0, :])
        hiddens = []
        for i in range(sequence.size()[1]):
            current_input = sequence[:, i, :]
            hidden = self.cln(current_input, hidden)
            hiddens.append(hidden)
        return torch.stack(hiddens, dim = 1), hidden
    
class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, context_type, visual_field):
        super().__init__()
        
        self.shaking_type = shaking_type
        self.visual_field = visual_field
        self.context_type = context_type
            
        if shaking_type == "cln":
            self.cond_layer_norm = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.tok_pair_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            self.context_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus2":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
            self.cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus3":
            self.cln_start_tok = LayerNorm(hidden_size, hidden_size, conditional = True)
            self.cln_end_tok = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "res_gate":
            self.Wg = nn.Linear(hidden_size, hidden_size)
            self.Wo = nn.Linear(hidden_size * 3, hidden_size)
            
        if context_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif context_type == "lstm":
            self.context_lstm = nn.LSTM(hidden_size, 
                           hidden_size, 
                           num_layers = 1, 
                           bidirectional = False, 
                           batch_first = True)
        elif context_type == "clngru":
            self.clngru = CLNGRU(hidden_size)
        elif context_type == "clnrnn":
            self.clnrnn = CLNRNN(hidden_size)
    
    def get_context_hiddens(self, seq_hiddens, context_type = "mean_pooling"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in context_type:
            context = torch.stack([pool(seq_hiddens[:, :i+1, :], context_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif context_type == "lstm":
            context, _ = self.context_lstm(seq_hiddens)
        elif context_type == "clngru":
            context, _ = self.clngru(seq_hiddens)
        elif context_type == "clnrnn":
            context, _ = self.clnrnn(seq_hiddens)
            
        return context
        
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: 
                shaking_seq_len: seq_len * visual_field - visual_field * (visual_field - 1) / 2
                cln: (batch_size, shaking_seq_len, hidden_size) (32, 5 * 3 - (2 + 1), 768)
                cat: (batch_size, shaking_seq_len, hidden_size * 2)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        padding_context = torch.zeros_like(seq_hiddens[:, 0,:])
        
        for start_idx in range(seq_len):            
            hidden_each_step = seq_hiddens[:, start_idx, :]
            # seq_len - start_idx: only shake afterwards
            repeat_len = min(seq_len - start_idx, self.visual_field)
            repeat_current_hiddens = hidden_each_step[:, None, :].repeat(1, repeat_len, 1)  
            visual_hiddens = seq_hiddens[:, start_idx:start_idx + self.visual_field, :]
            
            # context bt tok pairs
            context = self.get_context_hiddens(visual_hiddens, self.context_type)
#             context_list = []
#             for i in range(visual_hiddens.size()[1]):
#                 end_idx = start_idx + i
#                 ent_context = self.pooling(seq_hiddens[:, start_idx:end_idx + 1, :], self.context_type)
#                 bf_context = self.pooling(seq_hiddens[:, :start_idx, :], self.context_type) if start_idx != 0 else padding_context
#                 af_context = self.pooling(seq_hiddens[:, end_idx + 1:, :], self.context_type) if end_idx + 1 != seq_hiddens.size()[-2] else padding_context
#                 context_list.append((ent_context + bf_context + af_context) / 3)
#             context = torch.stack(context_list, dim = 1)
            
#             set_trace()
            if self.shaking_type == "cln":
                shaking_hiddens = self.cond_layer_norm(visual_hiddens, repeat_current_hiddens)
            elif self.shaking_type == "pooling":
                shaking_hiddens = context
            elif self.shaking_type == "cln_plus2":
                tok_pair_hiddens = torch.cat([repeat_current_hiddens, visual_hiddens], dim = -1)
                tok_pair_hiddens = torch.tanh(self.combine_fc(tok_pair_hiddens))
                shaking_hiddens = self.cln(context, tok_pair_hiddens)   
            elif self.shaking_type == "cln_plus":
                shaking_hiddens = self.tok_pair_cln(visual_hiddens, repeat_current_hiddens)
                shaking_hiddens = self.context_cln(shaking_hiddens, context)
            elif self.shaking_type == "cln_plus3":
                shaking_hiddens = self.cln_start_tok(context, repeat_current_hiddens)
                shaking_hiddens = self.cln_end_tok(shaking_hiddens, visual_hiddens)
            elif self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_current_hiddens, visual_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                shaking_hiddens = torch.cat([repeat_current_hiddens, visual_hiddens, context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "res_gate":
                gate = torch.sigmoid(self.Wg(repeat_current_hiddens))
                cond_hiddens = visual_hiddens * gate
                res_hiddens = torch.cat([repeat_current_hiddens, visual_hiddens, cond_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.Wo(res_hiddens)) 
            else:
                raise ValueError("Wrong shaking_type!")
                
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens
    
class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim = 0, center = True, scale = True, epsilon = None, conditional = False,
                 hidden_units = None, hidden_activation = 'linear', hidden_initializer = 'xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features = self.cond_dim, out_features = self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)

        self.initialize_weights()


    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢? 
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)


    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            
            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) **2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
