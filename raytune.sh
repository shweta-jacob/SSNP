#python ray_tuning.py --identifier hpo_neuro_test --output_path hpo_neuro_raytune --dataset hpo_neuro
#python ray_tuning.py --identifier hpo_metab_test --output_path hpo_metab_raytune --dataset hpo_metab
#python ray_tuning.py --identifier ppi_bp_test --output_path ppi_bp_raytune --dataset ppi_bp
#python ray_tuning.py --identifier em_user_test --output_path em_user_raytune --dataset em_user

python ray_tuning.py --identifier cut_ratio_test --output_path cut_ratio_raytune --dataset cut_ratio
python ray_tuning.py --identifier density_test --output_path density_raytune --dataset density
python ray_tuning.py --identifier coreness_test --output_path coreness_raytune --dataset coreness
python ray_tuning.py --identifier component_test --output_path component_raytune --dataset component