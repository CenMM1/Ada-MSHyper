import argparse
import torch
from exp.exp_main import Exp_Main

# Set random seed
torch.manual_seed(2024)

# Argument parser
parser = argparse.ArgumentParser(description='BimodalClassifier Multimodal Classification')

# =========================
#      Basic Config
# =========================
parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
# =========================
#      Data Config
# =========================
parser.add_argument('--root_path', type=str, default='./data', help='root path')
parser.add_argument('--num_workers', type=int, default=0, help='data loader workers')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

# =========================
#      Model Config
# =========================
parser.add_argument('--d_model', type=int, default=128, help='model dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--k', type=int, default=3, help='top-k hyperedges per node')

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
parser.add_argument('--loss_lambda', type=float, default=0.1, help='weight for hypergraph constraint loss')

args = parser.parse_args()

# GPU setup
args.use_gpu = torch.cuda.is_available()

print('Args:', args)

# Create experiment
Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = f'BimodalClassifier_multimodal_dm{args.d_model}_nc{args.num_classes}_{ii}'

        exp = Exp(args)
        print(f'>>> Start training: {setting} >>>')
        exp.train(setting)

        print(f'>>> Testing: {setting} <<<')
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    setting = f'BimodalClassifier_multimodal_dm{args.d_model}_nc{args.num_classes}_0'
    exp = Exp(args)
    exp.test(setting, test=1)

# Example usage:
# python run.py \
#   --is_training 1 \
#   --root_path ./datasets \
#   --d_model 128 \
#   --dropout 0.1 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 20 \
#   --patience 5 \
#   --hyper_num_text 50 \
#   --hyper_num_audio 20 \
#   --k 3 \
#   --num_classes 3 \
#   --seq_len_text 160 \
#   --seq_len_audio 518 \
#   --feature_dim_text 1024 \
#   --feature_dim_audio 1024 \
#   --loss_lambda 0.1
