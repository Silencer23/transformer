from atexit import register
from tokenize import Binnumber
from turtle import forward
import torch 
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import copy


# Positional Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # sin part, odd
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos part, even
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        '''
        Add persistent buffers to modules.
        This is typically used to register buffers that should not be considered model parameters. For example, pe is not a parameter, but part of a persistent state.
        Buffers can be accessed as properties with the given name.

        Description:
        Define a constant in memory that can be written and read while the model is saved and loaded
        '''
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) # cut to length of sentence
        return self.dropout(x)

# Attention Mechanism
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))

    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

# Multi Heads Attention Mechanism

def clones(moduel, n):
    return nn.ModuleList([copy.deepcopy(moduel) for _ in range(n)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim%head== 0
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask:
            mask = mask.unsqueeze(0) # extend dim to 4
        
        batch_size = query.size(0)
        # batch_size: batch_size
        # -1: length of sentence
        # self.head * self.d_k: embedding dims, each head get part of embedding attributes
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        return self.linears[-1](x)

# Feed Forward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

# Norm
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a2 * (x - mean) / (std + self.eps) + self.b2

# Connection
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # x + self.dropout(self.norm(sublayer(x)))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

#-------------------------- For test
# size = 512
# head = 8
# d_model = 512
# d_ff = 64
# c = copy.deepcopy
# attn = MultiHeadedAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# dropout = 0.2
# layer = EncoderLayer(size, c(attn), c(ff), dropout)
#
# Number of Encoder layers N
# N = 8
# mask = Variable(torch.zeros(8, 4, 4))
#
# en = Encoder(layer, N)
# en_result = en(x, mask)
# # print(en_result)
# # print(en_result.shape)


