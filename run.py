import argparse
import torch
from exp.exp_main import Exp_Main

# Set random seed
torch.manual_seed(2021)

# Argument parser
parser = argparse.ArgumentParser(description='ASHyper Time Series Forecasting')

# Basic config
parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
parser.add_argument('--model_id', type=str, default='weather_test', help='model id')
parser.add_argument('--model', type=str, default='ASHyper', help='model name')

# Data config
parser.add_argument('--data', type=str, default='weather', help='dataset type')
parser.add_argument('--root_path', type=str, default='./compressed_data', help='root path')
parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='M: multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='frequency')

# Model config
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--label_len', type=int, default=6, help='label length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction length')
parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
parser.add_argument('--c_out', type=int, default=21, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='model dimension')
parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='feed forward dimension')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# Training config
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

# Other config
parser.add_argument('--window_size', type=str, default=[2, 2])
parser.add_argument('--hyper_num', type=str, default=[50, 20, 10])
parser.add_argument('--num_workers', type=int, default=0, help='data loader workers')

args = parser.parse_args()

# GPU setup
args.use_gpu = torch.cuda.is_available()

print('Args:', args)

# Create experiment
Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_{ii}'

        exp = Exp(args)
        print(f'>>> Start training: {setting} >>>')
        exp.train(setting)

        print(f'>>> Testing: {setting} <<<')
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    setting = f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_0'
    exp = Exp(args)
    exp.test(setting, test=1)

# Example usage:
# python run.py --is_training 1 --model_id weather_final --model ASHyper --data weather --root_path ./compressed_data --data_path weather.csv --features M --target OT --seq_len 12 --label_len 6 --pred_len 12 --enc_in 21 --c_out 21 --d_model 16 --n_heads 1 --e_layers 1 --d_layers 1 --d_ff 64 --dropout 0.05 --batch_size 512 --learning_rate 0.1 --train_epochs 1 --itr 1 --num_workers 0
