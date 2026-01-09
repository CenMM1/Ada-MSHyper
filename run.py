import argparse
import torch
from exp.exp_main import Exp_Main

# Set random seed
import random
torch.manual_seed(random.randint(0, 10000))

# Argument parser
parser = argparse.ArgumentParser(description='MultimodalClassifier Multimodal Classification')

# =========================
#      Basic Config
# =========================
parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
# =========================
#      Data Config
# =========================
parser.add_argument('--root_path', type=str, default='./data', help='root path')
parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

# =========================
#      Model Config
# =========================
parser.add_argument('--d_model', type=int, default=128, help='model dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--k', type=int, default=3, help='top-k hyperedges per node')
parser.add_argument('--dynamic_hypergraph', type=int, default=1, help='1: dynamic hypergraph structure, 0: static')
parser.add_argument('--hyper_update_freq', type=int, default=5, help='update hypergraph every N steps to reduce computation')

# Hypergraph Attention
parser.add_argument('--hyper_heads', type=int, default=1,
                    help='number of attention heads in HypergraphConv (must divide d_model)')
parser.add_argument('--hyper_multi_head_attention', type=int, default=0,
                    help='use true multi-head hypergraph attention (1) or single-head (0)')

# Modality Specific Hyperparameters
parser.add_argument('--hyper_num_text', type=int, default=50, help='text modality hyperedges')
parser.add_argument('--hyper_num_audio', type=int, default=20, help='audio modality hyperedges')
parser.add_argument('--hyper_num_video', type=int, default=10, help='video modality hyperedges')

# =========================
#   Input Dimension Config
# =========================
# Sequence Lengths
parser.add_argument('--seq_len_text', type=int, default=160, help='text sequence length')
parser.add_argument('--seq_len_audio', type=int, default=518, help='audio sequence length')
parser.add_argument('--seq_len_video', type=int, default=16, help='video sequence length')

# Feature Dimensions
parser.add_argument('--feature_dim_text', type=int, default=1024, help='text feature dimension')
parser.add_argument('--feature_dim_audio', type=int, default=1024, help='audio feature dimension')
parser.add_argument('--feature_dim_video', type=int, default=2048, help='video feature dimension')

# =========================
#     Training Config
# =========================
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--kappa', type=float, default=0.1, help='weight for ECR regularization (paper: κ)')
# NOTE: legacy arg kept for backward compatibility; training loop no longer uses it.
parser.add_argument('--loss_lambda', type=float, default=0.1, help='[DEPRECATED] legacy weight (unused). Use --kappa instead')

args = parser.parse_args()

# GPU setup
args.use_gpu = torch.cuda.is_available()

print('Args:', args)

# Create experiment
Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = f'MultimodalClassifier_multimodal_dm{args.d_model}_nc{args.num_classes}_{ii}'

        exp = Exp(args)
        print(f'>>> Start training: {setting} >>>')
        exp.train(setting)

        print(f'>>> Testing: {setting} <<<')
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    setting = f'MultimodalClassifier_multimodal_dm{args.d_model}_nc{args.num_classes}_0'
    exp = Exp(args)
    exp.test(setting, test=1)

# Example usage:
# /opt/miniforge3/bin/conda run -n py385
# python run.py \
#   --is_training 1 \
#   --root_path ./preprocess/processed_data \
#   --num_workers 8 \
#   --num_classes 7 \
#   --d_model 128 \
#   --dropout 0.1 \
#   --k 3 \
#   --dynamic_hypergraph 1 \
#   --hyper_update_freq 5 \
#   --hyper_heads 1 \
#   --hyper_multi_head_attention 0 \
#   --hyper_num_text 50 \
#   --hyper_num_audio 20 \
#   --hyper_num_video 10 \
#   --seq_len_text 160 \
#   --seq_len_audio 518 \
#   --seq_len_video 16 \
#   --feature_dim_text 1024 \
#   --feature_dim_audio 1024 \
#   --feature_dim_video 2048 \
#   --batch_size 128 \
#   --learning_rate 0.0001 \
#   --train_epochs 15 \
#   --patience 3 \
#   --itr 1 \
#   --kappa 0.1


# python run.py \
#   --is_training 1 \
#   --root_path ./datasets \
#   --num_workers 8 \
#   --num_classes 7 \
#   --d_model 128 \
#   --dropout 0.1 \
#   --k 3 \
#   --hyper_heads 4 \
#   --hyper_multi_head_attention 1 \
#   --hyper_num_text 50 \
#   --hyper_num_audio 20 \
#   --hyper_num_video 10 \
#   --seq_len_text 160 \
#   --seq_len_audio 518 \
#   --seq_len_video 16 \
#   --feature_dim_text 1024 \
#   --feature_dim_audio 1024 \
#   --feature_dim_video 2048 \
#   --batch_size 100 \
#   --learning_rate 0.0001 \
#   --train_epochs 5 \
#   --patience 5 \
#   --itr 1 \
#   --kappa 0.1
