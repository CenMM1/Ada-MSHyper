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

📦 You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used directly.
# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
🚀 We provide the experiment scripts of Ada-MSHyper on all dataset under the folder `./scripts`. You can obtain the full results by running the following command:
```sh
```
or obtain specific results by runinng the following command:
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
We conduct extensive experiments to evaluate the performance and efficiency of Ada-MSHyper, covering long-range, short-range, and ultra-long-range time series forecasting, including 11 real-world benchmarks and 13 baselines.

**🏆 Ada-MSHyper achieves consistent state-of-the-art performance on all benchmarks**, covering a large variety of series with different frequencies, variate numbers and real-world scenarios.
## 5.1 Long-range forecasting
### 5.1.1 Long-range forecasting under multivariate settings.
![long-range-multivariate](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/long-range.png)
### 5.2 Long-range forecasting under univariate settings.
![long-range-univariate](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/longe-range-univariate.png)
## 5.2 Short-range forecasting
![short-range](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/short-range.png)
## 5.3 Ultra-long-range forecasting
![Ultra-long-range](https://github.com/shangzongjiang/Ada-MSHyper/blob/main/figures/Ultra-long-range.png)
