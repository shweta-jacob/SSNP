python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --samples 1 --m 8 --M 1 --stochastic --device 0
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --stochastic --device 1
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --stochastic --device 2
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --samples 1 --m 1 --M 5 --stochastic --device 3