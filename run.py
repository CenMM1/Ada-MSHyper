import argparse
import torch
from exp.exp_main import Exp_Main

# Set random seed
torch.manual_seed(2024)

# Argument parser
parser = argparse.ArgumentParser(description='BimodalClassifier Multimodal Classification')

# Basic config
parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
parser.add_argument('--model_id', type=str, default='multimodal_test', help='model id')
parser.add_argument('--model', type=str, default='BimodalClassifier', help='model name')

# Data config
parser.add_argument('--data', type=str, default='multimodal', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data', help='root path')
parser.add_argument('--data_path', type=str, default='multimodal_data', help='data file')

# Model config
parser.add_argument('--d_model', type=int, default=128, help='model dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
# HypergraphConv attention config (for ablation)
parser.add_argument('--hyper_heads', type=int, default=1,
                    help='number of attention heads in HypergraphConv (must divide d_model)')
parser.add_argument('--hyper_multi_head_attention', type=int, default=0,
                    help='use true multi-head hypergraph attention (1) or single-head (0)')

# Training config
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

# Model specific config
parser.add_argument('--hyper_num_text', type=int, default=50, help='text modality hyperedges')
parser.add_argument('--hyper_num_audio', type=int, default=20, help='audio modality hyperedges')
parser.add_argument('--hyper_num_video', type=int, default=10, help='video modality hyperedges')
parser.add_argument('--k', type=int, default=3, help='top-k hyperedges per node')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--num_workers', type=int, default=0, help='data loader workers')

args = parser.parse_args()

# GPU setup
args.use_gpu = torch.cuda.is_available()

print('Args:', args)

# Create experiment
Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = f'{args.model_id}_{args.model}_{args.data}_dm{args.d_model}_nc{args.num_classes}_{ii}'

        exp = Exp(args)
        print(f'>>> Start training: {setting} >>>')
        exp.train(setting)

        print(f'>>> Testing: {setting} <<<')
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    setting = f'{args.model_id}_{args.model}_{args.data}_dm{args.d_model}_nc{args.num_classes}_0'
    exp = Exp(args)
    exp.test(setting, test=1)

# Example usage:
# python run.py --is_training 1 --model_id multimodal_enhanced --model BimodalClassifier --data multimodal --root_path ./datasets --data_path train.pt --d_model 128 --dropout 0.05 --batch_size 128 --learning_rate 0.001 --train_epochs 5 --hyper_num_text 50 --hyper_num_audio 20 --k 3 --num_classes 7 --itr 1 --num_workers 0
