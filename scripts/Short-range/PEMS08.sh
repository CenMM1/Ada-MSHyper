
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=ASHyper

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.csv \
  --model_id PEMS08_$seq_len'_'96 \
  --CSCM Bottleneck_Construct \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'PEMS08_$seq_len'_'96.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.csv \
  --model_id PEMS08_$seq_len'_'192 \
  --CSCM Bottleneck_Construct \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'PEMS08_$seq_len'_'192.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.csv \
  --model_id PEMS08_$seq_len'_'336 \
  --CSCM Bottleneck_Construct \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'PEMS08_$seq_len'_'336.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.csv \
  --model_id PEMS08_$seq_len'_'720 \
  --CSCM Bottleneck_Construct \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'PEMS08_$seq_len'_'720.log