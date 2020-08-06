from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import time

class HandshakingKernel(nn.Module):
    def __init__(self, visual_field, fake_inputs, shaking_type):
        super().__init__()
        hidden_size = fake_inputs.size()[-1]
        
        self.shaking_type = shaking_type
        self.visual_field = visual_field
        
        if shaking_type == "cln":
            self.cond_layer_norm = LayerNorm(fake_inputs.size(), hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.lamtha = Parameter(torch.rand(hidden_size))
            self.tok_pair_cln = LayerNorm(fake_inputs.size(), hidden_size, conditional = True)
            self.context_cln = LayerNorm(fake_inputs.size(), hidden_size, conditional = True)
        elif shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "res_gate":
            self.Wg = nn.Linear(hidden_size, hidden_size)
            self.Wo = nn.Linear(hidden_size * 3, hidden_size)
    
    def mix_pooling(self, seq_hiddens):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        max_pooling, _ = torch.max(seq_hiddens, dim = -2)
        mean_pooling = torch.mean(seq_hiddens, dim = -2)
        mix_pooling = self.lamtha * mean_pooling + (1 - self.lamtha) * max_pooling
        return mix_pooling
        
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
            # full_hiddens4entity
#             context = torch.stack([self.mix_pooling(visual_hiddens[:, :i+1, :]) for i in range(visual_hiddens.size()[1])], dim = 1)
            
            context_list = []
            for i in range(visual_hiddens.size()[1]):
                end_idx = start_idx + i
                ent_context = self.mix_pooling(seq_hiddens[:, start_idx:end_idx + 1, :])
                bf_context = self.mix_pooling(seq_hiddens[:, :start_idx, :]) if start_idx != 0 else padding_context
                af_context = self.mix_pooling(seq_hiddens[:, end_idx + 1:, :]) if end_idx + 1 != seq_hiddens.size()[-2] else padding_context
                context_list.append((ent_context + bf_context + af_context) / 3)
            context = torch.stack(context_list, dim = 1)
            
#             set_trace()
            if self.shaking_type == "cln":
                shaking_hiddens = self.cond_layer_norm(visual_hiddens, repeat_current_hiddens)
            elif self.shaking_type == "full_hiddens":
                shaking_hiddens = full_ent_hiddens_mean
            elif self.shaking_type == "cln_plus":
                shaking_hiddens = self.tok_pair_cln(visual_hiddens, repeat_current_hiddens)
                shaking_hiddens = self.context_cln(shaking_hiddens, context)
            elif self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_current_hiddens, visual_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "res_gate":
                gate = torch.sigmoid(self.Wg(repeat_current_hiddens))
                cond_hiddens = visual_hiddens * gate
                res_hiddens = torch.cat([repeat_current_hiddens, visual_hiddens, cond_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.Wo(res_hiddens)) 
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens
    
class LayerNorm(nn.Module):
    def __init__(self, shape, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        shape: inputs.shape
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.shape = (shape[-1],)
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(self.shape))
        if self.scale:
            self.gamma = Parameter(torch.ones(self.shape))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=self.shape[0], bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=self.shape[0], bias=False)

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
            
            # cond在加入beta和gamma之前做一次变换，保证维度一致
            if self.center:
                beta = torch.sigmoid(self.beta_dense(cond)) + self.beta
            if self.scale:
                gamma = torch.sigmoid(self.gamma_dense(cond)) + self.gamma
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