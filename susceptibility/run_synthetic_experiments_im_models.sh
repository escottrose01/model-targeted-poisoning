#!/bin/bash

d2db=0.0
valid_theta_err=1.0
err_thresh=0.5
model_selection="min_collateral"
wdecay=5e-4

seps=($(seq 0.0 0.25 3.00))
flips=($(seq 0.0 0.1 1.0))
seeds=($(seq 1 10))
interps=($(seq 1.00 1.00))

mkdir -p "files/attack_anim/"
mkdir -p "files/attack_anim/model_selection-${model_selection}/"
mkdir -p "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/"
mkdir -p "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/"
mkdir -p "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/thresh${err_thresh}/"
for seed in "${seeds[@]}"; do
  for class_sep in "${seps[@]}"; do
    for flip_y in "${flips[@]}"; do
      mkdir -p "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/thresh${err_thresh}/sep${class_sep}-flip${flip_y}-seed${seed}"
      echo "running experiment with class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed}"

      # clear the stage
      rm -rf files/kkt_models \
        files/online_models \
        files/results files/target_classifiers \
        files/data/*_desc.csv \
        files/data/*_labels.txt \
        files/data/*_selected_subpops.txt

      # generate the dataset
      python generate_dataset.py --flip_y $flip_y --class_sep $class_sep --rand_seed $seed > /dev/null 2>&1
      if [ $? == 0 ]; then
        echo "generated dataset"
      else
        echo "dataset gen failed! exiting . . ."
        exit
      fi

      # run experiment on the dataset
      python generate_subpops.py --dataset synthetic --subpop_type cluster --num_subpops 16 > /dev/null 2>&1
      if [ $? == 0 ]; then
        echo "generated subpops"
      else
        echo "subpop gen failed! exiting . . ."
        exit
      fi

      # save copy of the subpops file
      cp files/data/synthetic_trn_cluster_desc.csv files/data/synthetic_trn_cluster_cpy_desc.csv

      for interp in "${interps[@]}"; do
        mkdir -p "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/thresh${err_thresh}/sep${class_sep}-flip${flip_y}-seed${seed}/interp${interp}"
        dst_fname="files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/thresh${err_thresh}/sep${class_sep}-flip${flip_y}-seed${seed}/interp${interp}/subpop_desc.csv"
        if !(test -f "${dst_fname}"); then
          # clear the stage (attack cache only)
          rm -rf files/kkt_models \
            files/online_models \
            files/results files/target_classifiers

          # restore empty subpop descs
          cp files/data/synthetic_trn_cluster_cpy_desc.csv files/data/synthetic_trn_cluster_desc.csv

          python generate_target_theta.py --dataset synthetic --model_type svm \
            --subpop_type cluster --weight_decay $wdecay --interp $interp --min_d2db $d2db \
            --valid_theta_err $valid_theta_err --selection_criteria $model_selection --all_subpops > /dev/null 2>&1
          if [ $? == 0 ]; then
            echo "generated target theta, interp=${interp}"
          else
            echo "target theta gen failed! exiting . . ."
            exit
          fi

          python run_kkt_online_attack.py --dataset synthetic --model_type svm \
            --subpop_type cluster --weight_decay $wdecay --require_acc --no_kkt \
            --target_model real --err_threshold $err_thresh --budget_limit 2000 \
            --target_valid_theta_err $valid_theta_err --sv_im_models > /dev/null 2>&1
          if [ $? == 0 ]; then
            echo "completed attack! interp=${interp}"; echo ""
          else
            echo "attack failed! exiting . . ."
            exit
          fi

          # move everything into a safe location
          mv files/online_models/synthetic/svm/cluster/12/orig/1/*.npz \
            "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/thresh${err_thresh}/sep${class_sep}-flip${flip_y}-seed${seed}/interp${interp}"
          cp "files/data/synthetic_train_test.npz" \
            "files/attack_anim/model_selection-${model_selection}/d2db${d2db}/valid_theta_err${valid_theta_err}/thresh${err_thresh}/sep${class_sep}-flip${flip_y}-seed${seed}/synthetic_train_test.npz"
          mv "files/data/synthetic_trn_cluster_desc.csv" $dst_fname
        else
          echo "experiment on class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed}, interp=${interp} already complete, skipping"
        fi
      done
    done
  done
done
