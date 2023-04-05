# SSNP: Stochastic Subgraph Neighborhood Pooling

SSNP (Stochastic Subgraph Neighborhood Pooling) is a fork of [GLASS](https://github.com/Xi-yuanWang/GLASS). SSNP uses subgraph neighborhoods during pooling to increase the expressiveness of plain-GNNs on subgraph classification.

#### Prepare Data

The realworld datasets can be downloaded from [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0).

### Commands to run our model
To run Online Views (OV): 
```
python ssnp.py --use_nodeid --use_seed --repeat 10 --epochs $epochs --dataset $dataset --model 2 --samples 1 --m $m --M $M --stochastic --device $gpu_id
```

To run Pre-processed Views (PV): 
```
python ssnp.py --use_nodeid --use_seed --repeat 10 --epochs $epochs --dataset $dataset --model 2 --samples 1 --m $m --M $M --views $views --device $gpu_id
```

To run Pre-processed Online Views (POV):
```
python ssnp_hybrid.py --use_nodeid --use_seed --repeat 10 --epochs $epochs --dataset $dataset --model 2 --m $m --M $M --nv $nv --nve $nve --device $gpu_id
```