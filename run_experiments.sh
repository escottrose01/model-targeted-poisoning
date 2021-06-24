#!/bin/bash

echo -e "dataset: "
read DATASET
echo -e "model type: "
read MODEL_TYPE
echo -e "subpop_type "
read SUBPOP_TYPE


python generate_subpops.py --dataset $DATASET --subpop_type $SUBPOP_TYPE
python generate_target_theta.py --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE

python run_kkt_online_attack.py --rand_seed 12 --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE
python run_kkt_online_attack.py --rand_seed 23 --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE
python run_kkt_online_attack.py --rand_seed 34 --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE
python run_kkt_online_attack.py --rand_seed 45 --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE

python process_avg_results.py --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE
python generate_table.py --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE
python plot_results.py --dataset $DATASET --model_type $MODEL_TYPE --subpop_type $SUBPOP_TYPE
