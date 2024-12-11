# <div align="center"> Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting
✨ This repo is the official implementation of Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting.
# 1 The framework of Ada-MSHyper
Ada-MSHyper is proposed to promote more comprehensive pattern interactions at different scales, which consist of four main parts: **Multi-scale Feature Extraction (MFE) Module**, **Adaptive Hypergraph Learning (AHL) Module**, **Multi-Scale Interaction Module**, and **Multi-Scale Fusion Module**. The overall framework of Ada-MSHyper is shown as follows:
![framework](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/framework.png)
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
# 3 Datasets && Description
![dataset-statistics](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/dataset%20statistics.png)
📦 You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used directly.
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
🚀 We provide the experiment scripts of Ada-MSHyper on all dataset under the folder `./scripts`. You can obtain the full results by running the following command:
```
# Train on ETTh1
sh ./scripts/Long-range/ETTh1.sh
# Train on ETTh2
sh ./scripts/Long-range/ETTh2.sh
# Train on ETTm1
sh ./scripts/Long-range/ETTm1.sh
# Train on ETTm2
sh ./scripts/Long-range/ETTm2.sh
# Train on Traffic
sh ./scripts/Long-range/traffic.sh
# Train on Electricity
sh ./scripts/Long-range/electricity.sh
# Train on Weather
sh ./scripts/Long-range/traffic.sh

```
or obtain specific results by runinng the following command:
```python
# Train on Electricity
python run_longExp.py -data elect -input_size 96 -predict_step 96 -root_path ./data/Electricity/ -data_path electricity.csv -CSCM Conv_Construct
# Train on ETTh1
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96_192 --model ASHyper --CSCM Bottleneck_Construct --data ETTh1 --features M --seq_len 96 --pred_len 192 --enc_in 7 --des 'Exp' --itr 1 --batch_size 16 --learning_rate 0.0001
# Train on ETTh2
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv --model_id ETTh2_96_96 --model ASHyper --CSCM Bottleneck_Construct --data ETTh2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.0001
# Train on ETTm1
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv --model_id ETTm1_96_96 --model ASHyper --CSCM Bottleneck_Construct --data ETTm1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 8 --learning_rate 0.0001
# Train on ETTm2
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv --model_id ETTm2_96_96 --model ASHyper --CSCM Bottleneck_Construct --data ETTm2 --features M --seq_len 96 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.001
# Train on Traffic
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path traffic.csv --model_id traffic_96_96 --model ASHyper --CSCM Bottleneck_Construct --data custom --features M --seq_len 96 --pred_len 96 --enc_in 862 --des 'Exp' --itr 1 --batch_size 16 --learning_rate 0.0001
# Train on Electricity
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path electricity.csv --model_id elect_96_96 --model ASHyper --CSCM Bottleneck_Construct  --data custom --features M --seq_len 96 --pred_len 96 --enc_in 321 --des 'Exp' --itr 1 --batch_size 16 --learning_rate 0.0001
# Train on Weather
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path weather.csv --model_id weather_96_96 --model ASHyper --CSCM Bottleneck_Construct --data custom --features M --seq_len 96 --pred_len 96 --enc_in 21 --des 'Exp' --itr 1 --batch_size 16 --learning_rate 0.0001
```
# 5 Main results
We conduct extensive experiments to evaluate the performance and efficiency of Ada-MSHyper, covering long-range, short-range, and ultra-long-range time series forecasting, including 11 real-world benchmarks and 13 baselines.

**🏆 Ada-MSHyper achieves consistent state-of-the-art performance on all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.
## 5.1 Long-range forecasting
### 5.1.1 Long-range forecasting under multivariate settings.
![long-range-multivariate](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/long-range.png)
![full-long-range-multivariate](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/full-long-multivariate.png)
### 5.2 Long-range forecasting under univariate settings.
![long-range-univariate](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/longe-range-univariate.png)
![full-long-range-univariate](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/full-long-univariate.png)
## 5.2 Short-range forecasting
![short-range](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/short-range.png)
![short-range](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/full-short.png)
## 5.3 Ultra-long-range forecasting
![Ultra-long-range](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/Ultra-long-range.png)
![full-Ultra-long-range](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/full-Ultra.png)
# Citation 
😀 If you find this repo useful, please cite our paper.
```
@inproceedings{shangada,
  title={Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting},
  author={Shang, Zongjiang and Chen, Ling and Wu, Binqing and Cui, Dongliang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}，
  year={2024}
}
```
# Concat
If you have any questions, please feel free to contact zongjiangshang@cs.zju.edu.cn
# Other works
📝 Our other works are shown as follows：

**Single-step forecasting**: Chen L, Chen D, Shang Z, et al. Multi-scale adaptive graph neural network for multivariate time series forecasting. TKDE, 2023, 35(10): 10748-10761.
[Code Link](https://github.com/shangzongjiang/MAGNN)

**AutoML related forecasting**: Chen D, Chen L, Shang Z, et al. Scale-aware neural architecture search for multivariate time series forecasting. TKDD, 2024. [Code Link](https://github.com/shangzongjiang/SNAS4MTF)

**Long-range time series forecasting**: Shang Z, Chen L, Wu B, et al. MSHyper: Multi-Scale Hypergraph Transformer for Long-Range Time Series Forecasting. arXiv, 2024: arXiv: 2401.09261
        
        . Code Link](https://github.com/shangzongjiang/MSHyper)




## The code and documentation are still being finalized, and the final version will be released after the conference.
