# repeat once for getting average inference and training times
python GLASSTest.py --use_nodeid --use_seed --repeat 1 --dataset ppi_bp --model 2 --samples 1 --m 8 --M 1 --stochastic --diffusion --device 0
python GLASSTest.py --use_nodeid --use_seed --repeat 1 --dataset hpo_metab --model 2 --samples 1 --m 1 --M 1 --stochastic --device 0
python GLASSTest.py --use_nodeid --use_seed --repeat 1 --dataset hpo_neuro --model 2 --samples 1 --m 1 --M 1 --stochastic --device 0
python GLASSTest.py --use_nodeid --use_seed --repeat 1 --dataset em_user --model 2 --samples 1 --m 1 --M 5 --stochastic --device 0