#!/bin/bash

# this script runs the visualization experiments against synthetic subpopulations. Each intermediate model induced by the online model-targeted attack algorithm is saved so that the attack process can be visualized.

# Dataset specification:
# - datasets are generated over a 13x11 grid of (class separation, label noise) dataset parameter pairs.
# - for each parameter combination, 10 different random seeds are used (seeds are reused between parameter combinations).

# Attack specification:
# - 16 cluster subpopulations are generated for each dataset.
# - target models are generated to achieve 100% test error on the target subpopulation, and selected by the criterion which minimizes loss on the non-subpopulation (collateral) clean data.
# - attack success is defined as at least 50% test error on target subpopulation.

valid_theta_err=1.0               # target model subpopulation error requirement
err_thresh=0.5                    # attack success requirement
model_selection="min_collateral"  # select target model candidate which minimizes collateral damage
wdecay=5e-4                       # model regularization parameter

seps=($(seq 0.0 0.25 3.00))       # class separation dataset parameter
flips=($(seq 0.0 0.1 1.0))        # label nois dataset parameter
seeds=($(seq 1 10))               # dataset seed

mkdir -p "files/attack_anim/"
mkdir -p "files/attack_anim/synthetic"
for seed in "${seeds[@]}"; do
  for class_sep in "${seps[@]}"; do
    for flip_y in "${flips[@]}"; do
      mkdir -p "files/attack_anim/synthetic/sep${class_sep}-flip${flip_y}-seed${seed}"
      dst_fname="files/attack_anim/synthetic/sep${class_sep}-flip${flip_y}-seed${seed}/subpop_desc.csv"
      echo "running experiment with class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed}"

      # clear the stage
      rm -rf files/kkt_models \
        files/online_models \
        files/results files/target_classifiers \
        files/data/*_desc.csv \
        files/data/*_labels.txt \
        files/data/*_selected_subpops.txt

      if !(test -f "${dst_fname}"); then
        # generate the dataset
        python generate_dataset.py --flip_y $flip_y --class_sep $class_sep --rand_seed $seed > /dev/null 2>&1
        if [ $? == 0 ]; then
          echo "generated dataset"
        else
          echo "dataset gen failed! exiting . . ."
          exit
        fi

        # generate subpopulations
        python generate_subpops.py --dataset synthetic --subpop_type cluster --num_subpops 16 > /dev/null 2>&1
        if [ $? == 0 ]; then
          echo "generated subpops"
        else
          echo "subpop gen failed! exiting . . ."
          exit
        fi

        # generate target models
        python generate_target_theta.py --dataset synthetic --model_type svm \
          --subpop_type cluster --weight_decay $wdecay \
          --valid_theta_err $valid_theta_err --selection_criteria $model_selection --all_subpops > /dev/null 2>&1
        if [ $? == 0 ]; then
          echo "generated target theta"
        else
          echo "target theta gen failed! exiting . . ."
          exit
        fi

        # run attack
        python run_kkt_online_attack.py --dataset synthetic --model_type svm \
          --subpop_type cluster --weight_decay $wdecay --require_acc --no_kkt \
          --target_model real --err_threshold $err_thresh --budget_limit 2000 \
          --target_valid_theta_err $valid_theta_err --sv_im_models > /dev/null 2>&1
        if [ $? == 0 ]; then
          echo "completed attack!"; echo ""
        else
          echo "attack failed! exiting . . ."
          exit
        fi

        # move everything into a safe location
        mv files/online_models/synthetic/svm/cluster/12/orig/1/*.npz \
          "files/attack_anim/synthetic/sep${class_sep}-flip${flip_y}-seed${seed}"
        cp "files/data/synthetic_train_test.npz" \
          "files/attack_anim/synthetic/sep${class_sep}-flip${flip_y}-seed${seed}/synthetic_train_test.npz"
        mv "files/data/synthetic_trn_cluster_desc.csv" $dst_fname
      else
        echo "experiment on class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed}, interp=${interp} already complete, skipping"
      fi

    done
  done
done
