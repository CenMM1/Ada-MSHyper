# <div align="center"> Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting
✨ This repo is the official implementation of Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting.
# 1 The framework of Ada-MSHyper
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
# 3 Datasets && Description

You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used directly.
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
```python
# Train on Weather
python train.py -data weather -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path weather.csv -CSCM Conv_Construct
# Train on Electricity
python train.py -data elect -input_size 168 -predict_step 168 -root_path ./data/Electricity/ -data_path electricity.csv -CSCM Conv_Construct
# Train on ETTh1
python train.py -data ETTh1 -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path ETTh1.csv -CSCM Conv_Construct
# Train on ETTm1
python train.py -data ETTm1 -input_size 168 -predict_step 168 -root_path ./data/ETT/ -data_path ETTm1.csv -CSCM Conv_Construct
# Train on Traffic
python train.py -data traffic -input_size 168 -predict_step 168 -root_path ./data/Traffic/ -data_path traffic.csv -CSCM Conv_Construct
```
# 5 Main results
