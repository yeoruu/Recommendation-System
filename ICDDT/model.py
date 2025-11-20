import torch.nn as nn
import torch
import math
from mydiff import DiffuRec
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
import os

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


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.beha_num = args.beha_num+1
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.beha_embeddings = nn.Embedding(self.beha_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.initializer_range = args.initializer_range
        
        self.model_name = str(self.__class__.__name__)
        self.save_path = args.output_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        fname = f"dataset_{args.dataset}_nl={args.n_layers}_nh={args.n_heads}_t={args.diffusion_steps}_gamma={args.gamma}_auxA={args.aux_alpha}_tarA={args.tar_alpha}_auxB={args.aux_beta}_tarB={args.tar_beta}_no={args.no}.pth"
        self.save_path = os.path.join(self.save_path, fname)
        
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
            
    def diffu_pre(self, item_rep, target_beha_rep, tag_emb, mask_seq, tag_beha_idx):
        seq_rep_diffu, item_rep_out, weights, t, seq_out  = self.diffu(item_rep, target_beha_rep, tag_emb, mask_seq, tag_beha_idx)
        return seq_rep_diffu, item_rep_out, weights, t, seq_out 

    def reverse(self, item_rep, target_beha_pre, noise_x_t, mask_seq, tag_beha_idx):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, target_beha_pre, noise_x_t, mask_seq, tag_beha_idx)
        return reverse_pre

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores, labels.squeeze(-1))
    
    def loss_seq_ce(self, seq_rep, labels):
        scores = torch.matmul(seq_rep, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))
    
    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores
    
    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)
        
        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))

    def forward(self, sequence, input_beha, tag, target_beha, train_flag=True): 
        print("\n ====== FORWARD DEBUG START ====== ")

        seq_length = sequence.size(1)
        print("sequence:", sequence.shape)
        print("input_beha:", input_beha.shape)
        print("target_beha", target_beha.shape)
        print("tag:", tag.shape)

        # --- Position embedding ---
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        print("position_embeddings:", position_embeddings.shape)

        # --- Item & Behavior embedding ---
        item_embeddings = self.item_embeddings(sequence)
        print("item_embeddings:", item_embeddings.shape)
        
        beha_embeddings = self.beha_embeddings(input_beha)
        print("input beha_embeddings:", beha_embeddings.shape)

        target_beha_embeddings = self.beha_embeddings(target_beha)
        print("target beha_embeddings:", target_beha_embeddings.shape)
        
        # Combined embedding
        item_embeddings = item_embeddings + position_embeddings + beha_embeddings
        print("combined item_embeddings:", item_embeddings.shape)

        # Dropout & LN
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm
        item_embeddings = self.LayerNorm(item_embeddings) # [B, L, D]
        print("after dropout+LN:", item_embeddings.shape)

        # Mask
        mask_seq = (sequence>0).float()
        print("mask_seq:", mask_seq.shape)

        tag_beha_idx = target_beha[:, -1] # [B, 1] # 마지막 행동을 기준으로 behavior-aware sampling
        print("tag_beha_idx:",tag_beha_idx.shape)

        if train_flag: # training phase
            tag_emb = self.item_embeddings(tag.squeeze(-1)) #+ self.beha_embeddings(tag_beha_idx)  ## B x H
            print("tag_emb:", tag_emb.shape)

            print("\n--- calling DIFFUSION model ---")
            rep_diffu, rep_item, weights, t, seq_rep = self.diffu_pre(item_embeddings, target_beha_embeddings, tag_emb, mask_seq, tag_beha_idx)
            item_rep_dis = None
            seq_rep_dis = None
            print("rep_diffu:", rep_diffu.shape)
            print("rep_item:", None if rep_item is None else rep_item.shape)
            print("weights:", weights.shape)
            print("t:", t.shape)
            print("seq_rep:", None if seq_rep is None else seq_rep.shape)

        else: # evaluation phase
            # noise_x_t = th.randn_like(tag_emb)
            print("\n--- INFERENCE MODE (reverse diffusion) ---")
            noise_x_t = th.randn_like(item_embeddings[:,-1,:])
            rep_diffu = self.reverse(item_embeddings, target_beha_embeddings, noise_x_t, mask_seq, tag_beha_idx)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None
            seq_rep = None 
            print("rep_diffu:", rep_diffu.shape)
        scores = None
        print("====== FORWARD DEBUG END ======\n")
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis, seq_rep
        

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre


