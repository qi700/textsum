import os
import torch
from numpy import random

root_dir = "../e-data/finished_files"
train_data_path = os.path.join(root_dir, "chunked/train_*")
eval_data_path = os.path.join(root_dir, "val.bin")
#decode_data_path = os.path.join(root_dir, "val.bin")
decode_data_path = os.path.join(root_dir, "chunked/val-test_*")
test_data_path = os.path.join(root_dir, "chunked/test_*")
vocab_path = os.path.join(root_dir, "vocab")
log_root = "./logs/e-data_adagrad"
save_model_path = './logs/e-data_adagrad/train_logs/models'
if not os.path.isdir(log_root):
    os.makedirs(log_root)

# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 16
# max_enc_steps = 400
max_enc_steps = 200
# max_dec_steps = 100
max_dec_steps = 40
beam_size = 4
# min_dec_steps = 35
min_dec_steps = 20
vocab_size = 50_000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
#pointer_gen = False
is_coverage = True
#is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 750_000

lr_coverage=0.15


intra_encoder = True
intra_decoder = True


# 使用GPU相关
use_gpu = True
GPU = "cuda:0"
USE_CUDA = use_gpu and torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device(GPU if USE_CUDA else 'cpu')

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)