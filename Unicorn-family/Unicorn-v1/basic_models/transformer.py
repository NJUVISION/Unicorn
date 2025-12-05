# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-07

import numpy as np
import torch
import MinkowskiEngine as ME
from pytorch3d.ops import knn_points, knn_gather


class Attention(torch.nn.Module):
    """Attention layer between two PCs.
    """
    def __init__(self, d_model=32, d_attn=16, num_heads=4, knn=16):
        super().__init__()
        self.knn = knn
        self.num_heads = num_heads
        assert d_model%self.num_heads==0 and d_attn%self.num_heads==0
        self.query_fn = torch.nn.Linear(d_model, d_attn)
        self.key_fn = torch.nn.Linear(d_model+3, d_attn)
        self.value_fn = torch.nn.Linear(d_model+3, d_model)

    def split_heads(self, x):
        """split last dimension to (num_heads, depth). 
            permute shape to (batch_size, num_heads, seq_len, depth)
        """
        if len(x.shape)==3:
            n, k, d = x.shape
            x = torch.reshape(x, (n, k, self.num_heads, d//self.num_heads))# [n,k,d]->[n,k,h,d/h]
            x = x.permute(0, 2, 1, 3)# [n,k,h,d/h] -> [n,h,k,d/h]
        if len(x.shape)==2:
            n, d = x.shape
            x = torch.reshape(x, (n, self.num_heads, d//self.num_heads))# [n,d]->[n,h,d/h]
        
        return x

    def forward(self, coords0, coords1, feats0, feats1, knn_idxs=None):
        """return feat1
        """
        coords0 = coords0.float().unsqueeze(0)# [1,n0,3]
        coords1 = coords1.float().unsqueeze(0)# [1,n1,3]

        if knn_idxs is None:
            dists, knn_idxs, _ = knn_points(coords1, coords0, K=self.knn, return_nn=False, return_sorted=True)

            assert knn_idxs.shape[1]==coords1.shape[1]
        
        # query, key, value
        # multi-head
        q = self.split_heads(self.query_fn(feats1))# [n1,h,d/h]
        k = self.split_heads(self.key_fn(torch.cat([knn_gather(feats0.unsqueeze(0), knn_idxs).squeeze(0), 
                                        (knn_gather(coords0, knn_idxs) - coords1.unsqueeze(-2)).squeeze(0)], dim=-1)))# [n1,h,k,d/h]
        v = self.split_heads(self.value_fn(torch.cat([knn_gather(feats0.unsqueeze(0), knn_idxs).squeeze(0), 
                                        (knn_gather(coords0, knn_idxs) - coords1.unsqueeze(-2)).squeeze(0)], dim=-1)))# [n1,h,k,d/h]

        # scaled dot-product attention
        attn = torch.einsum('nhd,nhkd->nhk', q, k)# [n,h,k]
        attn = attn/(k.shape[-1]**0.5)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        #
        # out = torch.einsum('nk,nkd->nd', attn, v)# [n1,d]
        out = torch.einsum('nhk,nhkd->nhd', attn, v)# [n,h,d/h]
        out = out.reshape(out.shape[0], -1)# [n,d]
        assert len(out)==len(feats1)

        return out, knn_idxs

    # def forward(self, coords0, coords1, feats0, feats1, knn_idxs=None):
    #     """return feat1
    #     """
    #     coords0 = coords0.float().unsqueeze(0)# [1,n0,3]
    #     coords1 = coords1.float().unsqueeze(0)# [1,n1,3]

    #     # knn_coords, knn_feats, _, knn_idxs = knn_fn(coords0, coords1, feats0, self.knn)
    #     if knn_idxs is None:
    #         dists, knn_idxs, _ = knn_points(coords1, coords0, K=self.knn, 
    #                                         return_nn=False, return_sorted=True)
    #         # assert (dists.cpu().squeeze()-((coords1.cpu().squeeze(0).unsqueeze(1).expand(-1,self.knn,-1) - \
    #         #     coords0.cpu().squeeze(0)[knn_idxs.squeeze()])**2).sum(dim=-1)).abs().min()<1e-3
    #         assert knn_idxs.shape[1]==coords1.shape[1]

    #     # relative positions
    #     knn_coords = knn_gather(coords0, knn_idxs)# [1,n1,k,3]
    #     knn_coords = knn_coords - coords1.unsqueeze(-2)# [1,n1,k,3]
    #     knn_coords = knn_coords.squeeze(0)# [n1,k,3]

    #     # print("DBG!! knn_coords", knn_coords.shape, knn_coords[0])
        
    #     # knn feature from pc0
    #     knn_feats = knn_gather(feats0.unsqueeze(0), knn_idxs).squeeze(0)# [n1,k,d]
    #     knn_feats = torch.cat([knn_feats, knn_coords], dim=-1)# [n1,k,d+3]
    #     # assert (knn_feats==feats0[knn_idxs.squeeze()]).all()
    #     # assert torch.abs(knn_feats - feats0[knn_idxs.squeeze()]).max()<1e-3
        
    #     # query, key, value
    #     q = self.query_fn(feats1)# [n1,d]
    #     k = self.key_fn(knn_feats)# [n1,k,d]
    #     v = self.value_fn(knn_feats)# [n1,k,d]
        
    #     # multi-head
    #     q = self.split_heads(q)# [n1,h,d/h]
    #     k = self.split_heads(k)# [n1,h,k,d/h]
    #     v = self.split_heads(v)# [n1,h,k,d/h]


    #     # scaled dot-product attention
    #     # attn = torch.einsum('nd,nkd->nk', q, k)# [n1,k]
    #     # attn = attn/(k.shape[-1]**0.5)
    #     # attn = torch.nn.functional.softmax(attn, dim=-1)
    #     attn = torch.einsum('nhd,nhkd->nhk', q, k)# [n,h,k]
    #     attn = attn/(k.shape[-1]**0.5)
    #     attn = torch.nn.functional.softmax(attn, dim=-1)
    #     #
    #     # out = torch.einsum('nk,nkd->nd', attn, v)# [n1,d]
    #     out = torch.einsum('nhk,nhkd->nhd', attn, v)# [n,h,d/h]
    #     out = out.reshape(out.shape[0], -1)# [n,d]
    #     assert len(out)==len(feats1)

    #     return out, knn_idxs


class Transformer(torch.nn.Module):
    def __init__(self, channels=32, knn=16):
        super().__init__()
        self.attention = Attention(d_model=channels, d_attn=channels, knn=knn)
        self.norm0 = torch.nn.LayerNorm(channels, eps=1e-5)
        self.linear = torch.nn.Linear(channels, channels)
        self.norm1 = torch.nn.LayerNorm(channels, eps=1e-5)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, coords0, coords1, feats0, feats1, knn_idxs):
        """input: Sparse Tensor with batch size of one.
        """
        out, knn_idxs = self.attention(coords0, coords1, feats0, feats1, knn_idxs)
        out = out + feats1
        out = self.norm0(out)
        #
        out = self.linear(out) + out
        out = self.norm1(out)

        return out, knn_idxs


class TransformerBlock(torch.nn.Module):
    def __init__(self, block_layers, channels, knn=16):
        super().__init__()
        self.linear0 =  torch.nn.Linear(channels, channels)
        self.layers = torch.nn.ModuleList()
        for i in range(block_layers):
            self.layers.append(Transformer(channels=channels, knn=knn))
        self.linear1 = torch.nn.Linear(channels, channels)

    def forward(self, x):
        """input: Sparse Tensor with batch size of one.
        """
        assert x.C[:,0].max()==0 
        # In the current stage, the batch size should to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        coords = x.C[:,1:].float() # [n,3]
        feats = x.F# [n,d]
        #
        feats = self.linear0(feats)
        knn_idxs = None
        for transformer in self.layers:
            feats, knn_idxs = transformer(coords, coords, feats, feats, knn_idxs)
        feats = self.linear1(feats)
        #
        out = ME.SparseTensor(feats,  
            coordinate_map_key=x.coordinate_map_key, 
            coordinate_manager=x.coordinate_manager, 
            device=x.device)
        
        return out