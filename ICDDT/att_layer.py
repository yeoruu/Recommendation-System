import torch.nn as nn
import math
import torch
import torch.nn.functional as F 


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.initializer_range = 0.02
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() 

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

# Multi-head Differential Attention
class MultiHeadDiffAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout, layer_idx):
        super().__init__()
        assert hidden_size % heads == 0
        self.heads = heads
        self.head_size = hidden_size // heads
        self.lambda_init = lambda_init(layer_idx) 

        # split qkv
        self.q1_proj = nn.Linear(hidden_size, hidden_size)
        self.q2_proj = nn.Linear(hidden_size, hidden_size)
        self.k1_proj = nn.Linear(hidden_size, hidden_size)
        self.k2_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, 2 * hidden_size)  # V projects to 2 * hidden_size

        self.c_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)
        self.subln = LayerNorm(2 * self.head_size)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(heads, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(heads, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(heads, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(heads, self.head_size) * 0.1)
        
        self.initializer_range = 0.02
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() 
            
    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape

        # Project x to get q1, q2, k1, k2, v
        q1 = self.q1_proj(q).view(B, T, self.heads, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(q).view(B, T, self.heads, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(k).view(B, T, self.heads, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(k).view(B, T, self.heads, self.head_size).transpose(1, 2)
        v = self.v_proj(v).view(B, T, self.heads, 2 * self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, att1.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,att1.shape[-1]])
            att1 = att1.masked_fill(mask == 0, -1e9) # float('-inf'))
            att2 = att2.masked_fill(mask == 0, -1e9) # float('-inf'))

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # [B, heads, T, 2 * head_size]
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, 2 * C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
        self.initializer_range = 0.02
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() 

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden



class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(AdaLayerNorm, self).__init__()
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return x

class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class ConditionAwareConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ConditionAwareConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.adnorm = AdaLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True) # scale & shift
        ) 
    def modulate(self, x, shift, scale):
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)
    
    def forward(self, x, sublayer, condition):
        if condition is not None:
            shift_k, scale_k = self.adaLN_modulation(condition).chunk(2, dim=1)
            return x + self.dropout((sublayer(self.modulate(self.adnorm(x), shift_k, scale_k))))
        else:
            return x + self.dropout(sublayer(self.norm(x)))

class FFNConditionAwareConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(FFNConditionAwareConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.adnorm = AdaLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 1 * hidden_size, bias=True)
        ) 
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, x, sublayer, condition):
        if condition is not None:
            gate_att = self.adaLN_modulation(condition)#.chunk(1, dim=1)
            return x + self.dropout(gate_att.unsqueeze(1) * sublayer(self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x)))


class AllTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, layer_idx):
        super(AllTransformerBlock, self).__init__()
        self.attention = MultiHeadDiffAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout, layer_idx=layer_idx)
        self.input_sublayer = ConditionAwareConnection(hidden_size=hidden_size, dropout=dropout)
        
        # B-SAL
        self.attention_beha = MultiHeadDiffAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout, layer_idx=layer_idx)
        self.feed_forward_beha = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer_beha = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer_beha = FFNConditionAwareConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, query, mask, condition):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask), condition)
        
        hidden = self.input_sublayer_beha(hidden, lambda _hidden: self.attention_beha.forward(query, _hidden, _hidden, mask=mask))
        
        hidden = self.output_sublayer_beha(hidden, self.feed_forward_beha, condition) 
        return self.dropout(hidden)

class All_Transformer_rep(nn.Module):
    def __init__(self, args):
        super(All_Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = args.n_heads
        self.dropout = args.dropout
        self.n_blocks = args.n_layers 
        self.all_transformer_blocks = nn.ModuleList(
            [AllTransformerBlock(self.hidden_size, self.heads, self.dropout, layer_idx+1) for layer_idx in range(self.n_blocks)])

    def forward(self, hidden, query, mask, condition=None):
        for transformer in self.all_transformer_blocks:
            hidden = transformer.forward(hidden, query, mask, condition)
        return hidden
    
    
    
    


























# 垃圾堆



class BehaTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, layer_idx):
        super(BehaTransformerBlock, self).__init__()
        # self.attention_beha = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.attention_beha = MultiHeadDiffAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout, layer_idx=layer_idx)
        self.feed_forward_beha = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer_beha = ConditionAwareConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer_beha = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, query, mask, condition):
        # LN -> ATT -> DP -> ADD -> LN -> FFN -> DP -> ADD
        # target behavior aware attention (as query)
        hidden = self.input_sublayer_beha(hidden, lambda _hidden: self.attention_beha.forward(query, _hidden, _hidden, mask=mask), condition)
        hidden = self.output_sublayer_beha(hidden, self.feed_forward_beha) 
        return self.dropout(hidden)

class Beha_Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Beha_Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = args.n_heads
        self.dropout = args.dropout
        self.n_blocks = args.n_layers # args.num_blocks
        self.beha_transformer_blocks = nn.ModuleList(
            [BehaTransformerBlock(self.hidden_size, self.heads, self.dropout, layer_idx+1) for layer_idx in range(self.n_blocks)])

    def forward(self, hidden, query, mask, condition=None):
        for transformer in self.beha_transformer_blocks:
            hidden = transformer.forward(hidden, query, mask, condition)
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout, layer_idx):
        super(TransformerBlock, self).__init__()
        # self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.attention = MultiHeadDiffAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout, layer_idx=layer_idx)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        # self.attention_beha = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        # self.feed_forward_beha = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        # self.input_sublayer_beha = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        # self.output_sublayer_beha = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, query, mask):
        # self-attention blocks
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        
        # # target behavior aware attention (as query)
        # hidden = self.input_sublayer_beha(hidden, lambda _hidden: self.attention_beha.forward(query, _hidden, _hidden, mask=mask))
        # hidden = self.output_sublayer_beha(hidden, self.feed_forward_beha) 
        
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = args.n_heads # 4
        self.dropout = args.dropout
        self.n_blocks = args.n_layers # args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout, layer_idx+1) for layer_idx in range(self.n_blocks)])

    def forward(self, hidden, query, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, query, mask)
        return hidden
    


class CrossAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        # 
        # self.q_lin = nn.Linear(hidden_size*2, hidden_size) 
        # self.k_lin = nn.Linear(hidden_size, hidden_size) 
        # self.v_lin = nn.Linear(hidden_size, hidden_size)
        
        # self.linear_layers = nn.ModuleList([self.q_lin, self.k_lin, self.v_lin])
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
        # TODO 先直接这样命令了
        self.initializer_range = 0.02
        
        self.apply(self._init_weights)
        
        self.dropout_output = nn.Dropout(p=dropout)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() 

    def forward(self, q, k, v, mask=None):
        # print(q.shape) # [B, 2D]
        # print(k.shape) # [B, L, D]
        key = k
        
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        
        hidden = key + self.dropout_output(hidden)
        return hidden