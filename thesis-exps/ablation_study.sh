# effect of diffusion
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --samples 1 --m 8 --M 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --stochastic --diffusion
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --stochastic --diffusion
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --samples 1 --m 1 --M 5 --stochastic --diffusion

# ablation study for non-stochasticity
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --samples 1 --m 8 --M 1 --views 5 --diffusion
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --views 5
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --views 5
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --samples 1 --m 1 --M 5 --views 5

# ablation study for subgraph only
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 0 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 0 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 0 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 0 --stochastic

# ablation study with sample size
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --samples 0.75 --m 8 --M 1 --stochastic --diffusion
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --samples 0.75 --m 1 --M 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --samples 0.75 --m 1 --M 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --samples 0.75 --m 1 --M 5 --stochastic

# ablation study with complement only
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 1 --stochastic

# ablation study with GCNConv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --samples 1 --m 8 --M 1 --stochastic --diffusion --use_gcn_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --stochastic --use_gcn_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --stochastic --use_gcn_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --samples 1 --m 1 --M 5 --stochastic --use_gcn_conv