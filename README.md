# Quant GANs replication
Codes for my thesis project: replicating and modifying Quant GANs.

[paper](https://arxiv.org/abs/1907.06673)

Python files for neural network creation/training, preprocessing and metrics can be found in the backend package. Models ought to be trained in Colab or using a PC with a CUDA capable GPU.

## Notebook files

Replication/testing:
* [S&P 500 Quant GAN training](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/sp500_training.ipynb)
* [Stylized facts of S&P 500 generated paths](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/stylized_facts_sp500.ipynb)
* [Mode collapse of Quant GANs on S&P 500 generated paths](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/mode_collapse_sp500.ipynb)
* [Training and testing for mode collapse of Quant GANS on the stocks compromising the Dow Jones index](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/train_dow_stocks.ipynb)
* [Generating multiple returns (MSFT/AAPL) with plausible dependencies](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/aapl_msft_train.ipynb)

`\Delta` CoVaR related:
* [Training conditionally on us publicly listed bank stocks](https://github.com/ICascha/QuantGANs-replication/blob/main/banking_train.ipynb)

Experiments found in the appendix:
* [Quant GANs using WGAN](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/appendix_experiments/wgan_sp500.ipynb)
* [TCN with series invariant filters comparison](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/appendix_experiments/series_invariant_filters.ipynb)
* [Researching TCN learning bias](https://nbviewer.jupyter.org/github/ICascha/QuantGANs-replication/blob/main/appendix_experiments/tcn_training_bias.ipynb)
