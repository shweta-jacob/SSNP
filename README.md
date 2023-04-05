# SSNP: Stochastic Subgraph Neighborhood Pooling

SSNP (Stochastic Subgraph Neighborhood Pooling) is a fork of [GLASS](https://github.com/Xi-yuanWang/GLASS). SSNP uses subgraph neighborhoods during pooling to increase the expressiveness of plain-GNNs on subgraph classification.

#### Prepare Data

The realworld datasets can be downloaded from [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0).

## Commands to run our model
### To run Online Views (OV): 
```
python ssnp.py --use_nodeid --use_seed --repeat 10 --epochs $epochs --dataset $dataset --model 2 --samples 1 --m $m --M $M --stochastic --device $gpu_id
```

### To run Pre-processed Views (PV): 
```
python ssnp.py --use_nodeid --use_seed --repeat 10 --epochs $epochs --dataset $dataset --model 2 --samples 1 --m $m --M $M --views $views --device $gpu_id
```

### To run Pre-processed Online Views (POV):
```
python ssnp_hybrid.py --use_nodeid --use_seed --repeat 10 --epochs $epochs --dataset $dataset --model 2 --m $m --M $M --nv $nv --nve $nve --device $gpu_id
```

## Command line arguments
The following command line arguments are supported by our model

- `dataset` - Set the dataset (currently supported datasets are ppi-bp, hpo-metab, hpo-neuro and em-user)
- `use_nodeid` - Use pretrained node embeddings (found in Emb folder)
- `repeat` - Set number of times experiment should be run
- `epochs` - Set the number of epochs for the model
- `m` - Set the length of the random walk (called h in the paper)
- `M` - Set the number of random walks (called k in the paper)
- `device` - Set the GPU ID to be used (eg; 0)
- `nv` - Set the total number of views
- `nve` - Set the number of views per epoch (only for POV)
- `diffusion` - An additional MLP after the pooling layer to mix the subgraph and neighborhood information

### Configs
The dataset configurations such as pooling function, learning rate, number of convolution layers etc., can be set in compl-config/$dataset.