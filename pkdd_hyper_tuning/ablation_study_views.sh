python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset ppi_bp --model 2 --samples 1 --m 1 --M 1 --diffusion --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --stochastic
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset em_user --model 2 --samples 1 --m 1 --M 1 --stochastic

python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset ppi_bp --model 2 --samples 1 --m 1 --M 1 --diffusion --views 5
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --views 5
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --views 5
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset em_user --model 2 --samples 1 --m 1 --M 1 --views 5

python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset ppi_bp --model 2 --samples 1 --m 1 --M 1 --diffusion --views 20
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --views 20
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --views 20
python GLASSTest.py --use_nodeid --use_seed --repeat 5 --dataset em_user --model 2 --samples 1 --m 1 --M 1 --views 20