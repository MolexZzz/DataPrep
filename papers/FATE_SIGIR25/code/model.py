import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from time import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
import subprocess  
import threading  
import time
import pickle
from torch import einsum
from einops import rearrange
import json
import os

# Transformer model
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            # print("pred", pred.shape)
            y_pred.append(pred)
        return y_pred

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # print("========================================")
        # print(f"Residual: {x.shape}")
        # print(f"Residual: {self.fn(x, **kwargs).shape}")
        # print("========================================")
        return self.fn(x, **kwargs) + x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # print("========================================")
        # print(f"PreNorm: {x.shape}")
        # print(f"PreNorm: {self.norm(x).shape}")
        # print(f"PreNorm: {self.fn(self.norm(x)).shape}")
        # print("========================================")
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

def choose_value_patch(attention, value, k):
    avg_attention = attention.mean(dim = 1)
    _, top_k_indices = avg_attention.topk(k, dim = -1)

    batch_size, _ = top_k_indices.shape
    batch_indices = torch.arange(batch_size)[:, None].to(top_k_indices.device)
    top_k_values = value[batch_indices, top_k_indices].view(batch_size, k , -1)

    return top_k_values

class First_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 16, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropput = nn.Dropout(dropout)
    
    def forward(self, x, key_padding_mask = None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)) # 按head进行划分

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        old_sim = sim

        if key_padding_mask is not None:
            array = key_padding_mask.cpu().numpy()
            all_true = np.all(array, axis=1)
            all_true_index = torch.tensor(all_true)

            key_padding_mask = key_padding_mask.bool()
            key_padding_mask = ~key_padding_mask

            sim = sim.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

            sim = sim.float()
            sim[all_true_index] = old_sim[all_true_index]

        attn = sim.softmax(dim = -1)
        attn = attn.float()
        # Apply attention weights to values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # Concatenate heads and apply output layer
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 16, dropout = 0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class Last_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 16, dropout = 0, batch_size = 256, seq_len = 17, device = "cuda"):
        super(Last_Attention, self).__init__()
        self.p_dim = 2
        inner_dim = heads * dim_head
        self.heads = heads
        self.temperature = 1
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.dim_head = dim_head
        self.inner_dim = inner_dim

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.momentum = 0.1
        self.register_buffer('running_mean_q', torch.zeros(self.batch_size, self.heads, self.seq_len, self.dim_head))
        self.register_buffer('running_std_q', torch.ones(self.batch_size, self.heads, self.seq_len, self.dim_head))
        self.register_buffer('running_mean_k', torch.zeros(self.batch_size, self.heads, self.seq_len, self.dim_head))
        self.register_buffer('running_std_k', torch.ones(self.batch_size, self.heads, self.seq_len, self.dim_head))

        self.projector = nn.Sequential(
            nn.Linear(self.p_dim * inner_dim, 512, bias = False),
            nn.ReLU(),
            nn.Linear(512, 128, bias = False)
        )

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def forward(self, x, training):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)) # 按head进行划分

        self.running_mean_q = self.running_mean_q.detach()
        self.running_std_q = self.running_std_q.detach()
        self.running_mean_k = self.running_mean_k.detach()
        self.running_std_k = self.running_std_k.detach()
        
        if training:
            q_mean, q_std = torch.mean(q, 0, keepdim=True), torch.std(q, 0, keepdim=True)
            k_mean, k_std = torch.mean(k, 0, keepdim=True), torch.std(k, 0, keepdim=True)  

            self.running_mean_q = (1 - self.momentum) * self.running_mean_q.to(self.device) + self.momentum * q_mean
            self.running_std_q = (1 - self.momentum) * self.running_std_q.to(self.device) + self.momentum * q_std
            self.running_mean_k = (1 - self.momentum) * self.running_mean_k.to(self.device) + self.momentum * k_mean
            self.running_std_k = (1 - self.momentum) * self.running_std_k.to(self.device) + self.momentum * k_std
        else:
            with torch.no_grad():
                q_mean = self.running_mean_q.to(self.device)
                q_std = self.running_std_q.to(self.device)
                k_mean = self.running_mean_k.to(self.device)
                k_std = self.running_std_k.to(self.device)
                    
                
        q = (q - q_mean)  / q_std
        q = torch.abs(q)
        k = (k - k_mean) / k_std
        k = torch.abs(k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        attentions = attn[:,:, 0, :]
        v = v.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.inner_dim)
        mst_val = choose_value_patch(attentions, v, self.p_dim)
        mst_val = mst_val.reshape(self.batch_size, -1)
        mst_val = self.projector(mst_val)
        z = F.normalize(mst_val, dim=1)
        return out

class Last_ATBlock(nn.Module):
    def __init__(self, normalize):
        super().__init__()
        dim = 768
        self.norm = nn.LayerNorm(dim)
        self.attention = Last_Attention(normalize)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x, training):
        identity = x
        x = self.norm(x)
        x, vz, att = self.attention(x, training)
        x += identity
        res = x 
        x = self.norm2(x)
        x = self.feedforward(x)
        x += res
        return x, vz, att

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, n_feat, batch_size,
                 mask_missing = False, normalize_k_q = False, device = "cuda"):
        super().__init__()
        
        self.mask_missing = mask_missing
        self.normalize_k_q = normalize_k_q
        self.layers = nn.ModuleList([])
        self.attn = nn.MultiheadAttention(dim, heads, attn_dropout)
        self.depth = depth
        for n_layer in range(depth):
            if n_layer == 0:
                if mask_missing:
                    self.layers.append(nn.ModuleList([
                        PreNorm(dim, Residual(First_Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                        PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    ]))
                else:
                    self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]))
            elif n_layer == depth - 1:
                if self.normalize_k_q:
                    self.layers.append(nn.ModuleList([
                        PreNorm(dim, Residual(Last_Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, batch_size = batch_size, seq_len = n_feat, device = device))),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]))
                else:
                    self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]))

            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                ]))
        
    def forward(self, x_cat, x_cont = None, cont_mask = None, cat_mask = None, training = False):
        if cont_mask is not None:
            cont_mask = torch.cat((cat_mask, cont_mask), dim = 1).bool()

        if x_cont is not None:
            x = torch.cat((x_cat, x_cont), dim=1)
            
        _, n, _ = x.shape

        n_layer = 0
        for attn1, ff1 in self.layers:
            if n_layer == 0:
                if self.mask_missing:
                    x = attn1(x, key_padding_mask = cont_mask)
                else:
                    x = attn1(x)
            elif n_layer == self.depth - 1:
                if self.normalize_k_q:
                    x = attn1(x, training = training)
                else:
                    x = attn1(x)
            else:
                x = attn1(x)

            x = ff1(x)
            n_layer += 1

        return x


class FATE(nn.Module):
    def __init__(self, categories, num_continuous, dim, depth, heads, dim_head = 16, dim_out = 1,
                 mlp_hidden_mults = (4, 2), mlp_act = None, num_special_tokens = 0, attn_dropout = 0,
                 ff_dropout = 0, cont_embeddings = "MLP", attentiontype = 'col', batch_size = 256,
                 final_mlp_style = 'common', y_dim = 2, device = "cuda"):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        self.batch_size = batch_size
        self.num_categories = len(categories) # categories的列的数量
        self.num_unique_categories = sum(categories) # categories所对应的unique值的数量

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        # 对于连续型变量的处理
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        mask_missing = False
        normalize_k_q = False
        if attentiontype == "col":
            mask_missing = False
            normalize_k_q = False
        elif attentiontype == "mask":
            mask_missing = True
            normalize_k_q = False
        elif attentiontype == "normalize":
            mask_missing = False
            normalize_k_q = True
        elif attentiontype == "mask_normalize":
            mask_missing = True
            normalize_k_q = True
        # elif attentiontype == "row":
        # elif attentiontype == "rowcol":

        self.transformer = Transformer(
            num_tokens = self.total_tokens,
            dim = dim, 
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            n_feat = nfeats,
            mask_missing = mask_missing,
            normalize_k_q = normalize_k_q,
            batch_size = batch_size,
            device = device
        )
        
            
        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) #左侧填1列为0
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        
        # 构建将embedding后的分类数据和连续数据恢复原有形式的mlp
        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            # print(f"1111self.num_categories: {self.num_categories}")
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

    def forward(self, x_categ, x_cont, cont_mask, cat_mask, training):
        if cont_mask == None:
            x = self.transformer(x_categ, x_cont, training)
        else:
            x = self.transformer(x_categ, x_cont, cont_mask, cat_mask, training)
        cat_outs = self.mlp1(x[:,:self.num_categories,:])
        con_outs = self.mlp2(x[:,self.num_categories:,:])
        return cat_outs, con_outs  