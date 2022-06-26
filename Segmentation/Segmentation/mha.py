import torch
import numpy as np
from torch import nn
import math

from Segmentation.common.consts import DEVICE
from Segmentation.common.utils import position_encoding


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.fc1 = nn.Linear(5*128,256) # 10 
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1) 
        
    def forward(self,din):
        # din = din.view(-1,500)
        # print(din)
        dout = nn.functional.relu(self.fc1(din))
        dout = nn.functional.relu(self.fc2(dout))
        dout = nn.functional.relu(self.fc3(dout))
        return self.fc4(dout), torch.sigmoid(self.fc4(dout)) 


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads=8,
            embed_dim=128,
            n_layers=2,
            node_dim=5+5,  
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim)  #if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

        self.classify = Net2()

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        batch_size, node_size, dim_size = x.shape  # (5*500, 10, 5)

        # positional encoding
        # print(f"In SeqNet forward: x: {x.shape}")
        position_enc = position_encoding(node_size, 5).to(DEVICE)  # (10,5)
        pe = torch.stack([position_enc for _ in range(batch_size)], dim=0) # (2500, 10, 5)
        x = torch.cat([x, pe], dim=-1)  # (2500, 10, 5+5) 
        # print(f"In SeqNet forward: x with position encoding: {x.shape}")  

        # Batch multiply to get initial embeddings of nodes 
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) #if self.init_embed is not None else x

        h = self.layers(h)  # (2500, 10, 128)

        # print(f"In SeqNet forward: initial return h: {h.shape}")
        a,b,c = h.shape  # (2500, 10, 128)
        # print(f"h: {h.shape}")
        h = h.reshape(a,b*c) #  (2500, 10*128)
        # print(f"In SeqNet forward: reshaped return h: {h.shape}")
        h, sig_h = self.classify(h)  #(2500, 1)
        # print(f"In SeqNet forward: classified return h: {h.shape}, {sig_h.shape}")

        return h, sig_h 
        # (
        #     h,  # (batch_size, graph_size, embed_dim)
        #     h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        # )
    
class MLPEncoder(nn.Module):
    def __init__(
            self,
            n_heads=8,
            embed_dim=128,
            n_layers=2,
            node_dim=5+5,  
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(MLPEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim)  #if node_dim is not None else None

        # self.layers = nn.Sequential(*(
        #     MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
        #     for _ in range(n_layers)
        # ))

        self.classify = Net2()

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        batch_size, node_size, dim_size = x.shape  # (5*500, 20, 5)

        # positional encoding
        # print(f"In SeqNet forward: x: {x.shape}")
        position_enc = position_encoding(node_size, 5).to(DEVICE)  # (10,5)
        pe = torch.stack([position_enc for _ in range(batch_size)], dim=0) # (2500, 10, 5)
        x = torch.cat([x, pe], dim=-1)  # (2500, 10, 5+5) 
        # print(f"In SeqNet forward: x with position encoding: {x.shape}")  

        # Batch multiply to get initial embeddings of nodes 
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) #if self.init_embed is not None else x

        # h = self.layers(h)  # (2500, 10, 128)

        # print(f"In SeqNet forward: initial return h: {h.shape}")
        a,b,c = h.shape  # (2500, 10, 128)
        h = h.reshape(a,b*c) #  (2500, 10*128)
        # print(f"In SeqNet forward: reshaped return h: {h.shape}")
        h, sig_h = self.classify(h)  #(2500, 1)
        # print(f"In SeqNet forward: classified return.  h: {h.shape}, {sig_h.shape}")

        return h, sig_h 
        

class Net_fixing(nn.Module):
    def __init__(self):
        super(Net_fixing,self).__init__()
        self.fc1 = nn.Linear(20*128,256) # 10 
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,11) 
        
    def forward(self,din):
        # din = din.view(-1,500)
        # print(din)
        dout = nn.functional.relu(self.fc1(din))
        dout = nn.functional.relu(self.fc2(dout))
        dout = nn.functional.relu(self.fc3(dout))
        return self.fc4(dout), torch.sigmoid(self.fc4(dout)) 

class FixingMLPEncoder(nn.Module):
    def __init__(
            self,
            n_heads=8,
            embed_dim=128,
            n_layers=2,
            node_dim=5+5,  
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(FixingMLPEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim)  #if node_dim is not None else None

        # self.layers = nn.Sequential(*(
        #     MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
        #     for _ in range(n_layers)
        # ))

        self.classify = Net_fixing()

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        batch_size, node_size, dim_size = x.shape  # (5*500, 20, 5)

        # positional encoding
        # print(f"In SeqNet forward: x: {x.shape}")
        position_enc = position_encoding(node_size, 5).to(DEVICE)  # (20,5)
        pe = torch.stack([position_enc for _ in range(batch_size)], dim=0) # (2500, 20, 5)
        x = torch.cat([x, pe], dim=-1)  # (2500, 20, 5+5) 

        # Batch multiply to get initial embeddings of nodes 
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) #if self.init_embed is not None else x
        # (2500, 20, 5+5) -> (2500, 20, 128)

        # h = self.layers(h)  # (2500, 20, 128)

        a,b,c = h.shape  # (2500, 20, 128)
        h = h.reshape(a,b*c) #  (2500, 20*128)
        h, sig_h = self.classify(h)  #(2500, 1)

        return h, sig_h 
        