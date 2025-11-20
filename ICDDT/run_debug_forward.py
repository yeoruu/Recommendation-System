import torch 
from model import create_model_diffu, Att_Diffuse_model
import argparse

# 1) dummy args 생성 
args = argparse.Namespace(
    hidden_size=128,
    item_num=100,
    beha_num=5,
    max_seq_length=10,

    emb_dropout=0.1,
    dropout=0.1,
    initializer_range=0.02,

    output_dir='./debug_save/',
    dataset='debug',

    # Transformer
    n_layers=2,
    n_heads=2,

    # Diffusion
    diffusion_steps=4,
    gamma=1.0,
    aux_alpha=1.0,
    tar_alpha=1.0,
    aux_beta=1.0,
    tar_beta=1.0,
    schedule_sampler_name="behaaware",
    lambda_uncertainty=0.001,
    noise_schedule="trunc_lin",
    rescale_timesteps=True,

    # other
    no=0
)


# 2) diffusion 모델 생성 
diffu = create_model_diffu(args)

# 3) 전체 모델 생성 
model = Att_Diffuse_model(diffu, args)

# 4) dummy input 만들기 
B = 1 # batch 
L = 5 # sequence length 

sequence = torch.randint(1, 30, (B, L))
input_beha = torch.randint(1, 3, (B, L))
target_beha = torch.randint(1, 3, (B, L))
tag = torch.randint(1, 30, (B, 1))

# 5) forward 디버그
print("Running forward debug...")
model.forward(sequence, input_beha, tag, target_beha, train_flag=True)