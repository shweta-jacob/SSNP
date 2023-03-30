# gat, sage on all datasets

# ppi-bp
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --m 1 --M 1 --views 5 --diffusion --device 0 --use_gat_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset ppi_bp --model 2 --m 1 --M 1 --views 5 --diffusion --device 0 --use_sage_conv

# hpo-metab
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --m 1 --M 5 --views 5 --diffusion --device 0 --use_gat_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_metab --model 2 --m 1 --M 5 --views 5 --diffusion --device 0 --use_sage_conv

# hpo-neuro
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --m 1 --M 1 --views 5 --device 0 --use_gat_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset hpo_neuro --model 2 --m 1 --M 1 --views 5 --device 0 --use_sage_conv

# em-user
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --m 1 --M 1 --views 5 --device 0 --use_gat_conv
python GLASSTest.py --use_nodeid --use_seed --repeat 10 --dataset em_user --model 2 --m 1 --M 1 --views 5 --device 0 --use_sage_conv
