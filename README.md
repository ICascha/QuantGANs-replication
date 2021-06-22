# Quant GANs replication
Codes for my thesis project: replicating and modifying Quant GANs.

[paper](https://arxiv.org/abs/1907.06673)

Python files for neural network creation/training, preprocessing and metrics can be found in the backend package.

## Notebook files

Replication/testing:
* [S&P 500 Quant GAN training](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/sp500_training.ipynb)
* [Stylized facts of S&P 500 generated paths](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/stylized_facts_sp500.ipynb)
* [Mode collapse of Quant GANs on S&P 500 generated paths](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/mode_collapse_sp500.ipynb)
* [Training and testing for mode collapse of Quant GANS on the stocks compromising the Dow Jones index](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/train_dow_stocks.ipynb)

Personal modifications/approcahes to Quant GANS:
* [Generating multiple returns (MSFT/AAPL) with plausible cross-autocorrelations](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/aapl_msft_train.ipynb)

Experiments found in the appendix:
* [Quant GANs using WGAN](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/appendix_experiments/series_invariant_filters.ipynb)
* [TCN with series invariant filters comparison](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/appendix_experiments/series_invariant_filters.ipynb)
* [Researching TCN learning bias](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/appendix_experiments/series_invariant_filters.ipynb)


## TCN using 2 dimensional convolutions
I modify a standard TCN with skip layers to use 2d convoltion layers. On the temporal axis we use causal dialated convolutions, and on the other axis we differentiate multiple time-series. Inside a temporal block, we split each convolution layer in multiple convolution layers (depending on the amount of time-series modeled) and concatonate thereafter in order to capture cross dependencies between time-series while having seperate weights for each time-series. (See picture for a stylized example, showing computation of just 2 time-steps to avoid clutter).

![Image of 2d TCN architecture](https://github.com/ICascha/QuantGANs-replication/blob/main/images/conv2d_tcn.png?raw=true)
