from atexit import register
from pyexpat import model
from tokenize import Binnumber
from turtle import forward, position
import torch 
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import copy

# Embedding 
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model


    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

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


# ------------------------------------- Decoder

# Decoder Layer

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout) -> None:
        super(DecoderLayer, self).__init__()
        # size: embedding dim
        # self_attn: object of Multiple-head attention
        # src_attn: object of normal attention
        # feed_forward: object of feed forward connection
        # dropout: dropout

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # clone 3 connection
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, source_mask, target_mask):
        # x: output of last layer
        # memory: Semantic storage variables from the encoder layer
        # source_mask: source data mask
        # target_mask: target data mask

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))

        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Decoder, self).__init__()
        # N: clone N decoder layers

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, source_mask, target_mask):
        # x: output of last layer
        # memory: Semantic storage variables from the encoder layer
        # source_mask: source data mask
        # target_mask: target data mask

        # loop N times
        # norm results
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        
        return self.norm(x)

# -----------------------------for test
# # N: num layers
# size = 512
# d_model = 512
# head = 8
# d_ff = 64
# dropout = 0.2
# c = copy.deepcopy
# attn = MultiHeadedAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
# N = 8
# # input = encoder
# x = pe_result
# memory = en_result
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask
# de = Decoder(layer, N)
# de_result = de(x, memory, source_mask, target_mask)
# # print(de_result)
# # print(de_result.shape)

# Output part

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)

# Encoder - Decoder

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_emb, target_emb, generator) -> None:
        super(EncoderDecoder, self).__init__()
        # encoder, decoder object
        # source and target embedding func
        # output generator object
        self.encoder = encoder
        self.decoder = decoder
        self.source_emb = source_emb
        self.target_emb = target_emb
        self.generator = generator

    def forward(self, source, target, source_mask, targte_mask):
        return self.generator(self.decode(self.encode(source, source_mask), source_mask,
                                        target, targte_mask))

    def encode(self, source, source_mask):
        return self.encoder(self.source_emb(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        # memory: output of encoder
        return self.decoder(self.target_emb(target), memory, source_mask, target_mask)

# Build model
def build_model(source_vocab, target_vocab, N=6,
                d_model=512, d_ff=2048, head=8, dropout=0.2):
    
    # source_vocab, target_vocab: length of vacob
    # N: stacks number of encoder, decoder
    # d_model: embedding dim
    # d_ff: feed forward connection dim
    # head: number of muti-head
    # dropout: dropout

    # rename deepcopy
    c = copy.deepcopy

    attn = MultiHeadedAttention(head, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    # Construct Model
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    '''
    Once the model structure is complete, the next step is to initialize the parameters in the model, such as the transformation matrix in the linear layer
    Once the dimension of the judged parameter is greater than 1, it will be initialized to a matrix subject to uniform distribution.
    '''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model

# test
source_vocab = 11
target_vocab = 11
N = 6
# use default parameters
if __name__ == '__main__':
    model = build_model(source_vocab, target_vocab, N)
    print(model)