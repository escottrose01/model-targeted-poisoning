#!/bin/bash

# this script runs the lower-bound estimate poisoning experiments against synthetic subpopulations. Intermediate subpopulations are not saved.

# Dataset specification:
# - datasets are generated over a 4x4 grid of (class separation, label noise) dataset parameter pairs.
# - for each parameter combination, 10 different random seeds are used (seeds are reused between parameter combinations).

# Attack specification:
# - 16 cluster subpopulations are generated for each dataset.
# - target models are generated to achieve 100% test error on the target subpopulation, and selected by the criterion which minimizes loss on the non-subpopulation (collateral) clean data.

valid_theta_err=0.5               # target model subpopulation error requirement
err_thresh=0.5                    # attack success requirement
model_selection="min_collateral"  # select target model candidate which minimizes collateral damage
wdecay=5e-4                       # model regularization parameter

seps=($(seq 0.0 0.25 3.00))        # class separation dataset parameter
flips=($(seq 0.0 0.1 1.0))        # label noise dataset parameter
seeds=($(seq 1 10))               # dataset seed

mkdir -p "files/out/synthetic_lowerbound"
for seed in "${seeds[@]}"; do
  for class_sep in "${seps[@]}"; do
    for flip_y in "${flips[@]}"; do
      mkdir -p "files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}"
      dst_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/subpop_desc.csv"
      sv_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/subpop_desc_tmp.csv"
      subpop_fname="files/data/synthetic_svm_cluster_selected_subpops.txt"

      if !(test -f "${dst_fname}"); then
        echo "running experiment with class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed}"

        # clear the stage
        rm -rf files/kkt_models \
          files/online_models \
          files/results files/target_classifiers \
          files/data/*_desc.csv \
          files/data/*_labels.txt \
          files/data/*_selected_subpops.txt

        # generate the dataset
        python generate_dataset.py --flip_y $flip_y --class_sep $class_sep --rand_seed $seed > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "generated dataset"
        else
          echo "dataset gen failed! exiting . . ."
          exit
        fi

        # generate subpopulations
        python generate_subpops.py --dataset synthetic --lazy --subpop_type cluster --num_subpops 16 > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "generated subpops"
        else
          echo "subpop gen failed! exiting . . ."
          exit
        fi

        # generate target models
        python generate_target_theta.py --dataset synthetic --model_type svm \
          --subpop_type cluster --weight_decay $wdecay --save_all \
          --valid_theta_err $valid_theta_err --selection_criteria $model_selection --all_subpops > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "generated target theta"
        else
          echo "target theta gen failed! exiting . . ."
          exit
        fi

        # compute lower bounds for subpop attacks
        python generate_subpop_lowerbounds.py --model_type svm --dataset synthetic \
          --weight_decay $wdecay --subpop_type cluster --valid_theta_err $valid_theta_err \
          > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "computed lower bounds"
        else
          echo "lower bound computation failed! exiting . . ."
          exit
        fi

        # recover preexisting .csv data, if exists
        if (test -f "${sv_fname}"); then
          cp $sv_fname "files/data/synthetic_trn_cluster_desc.csv"
        fi

        # iterate over all subpops (individually)
        read -r -a subpops < $subpop_fname
        echo -n "completed attacks against subpops:"
        for subpop_id in "${subpops[@]}"; do
          # check if attack is already completed against this subpop
          atk_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/subpop-${subpop_id}-atk.npz"
          if !(test -f "${atk_fname}"); then
            # run attack
            python susceptibility/run_lowerbound_experiments.py --dataset synthetic \
                --model_type svm --subpop_type cluster --weight_decay $wdecay \
                --require_acc --err_threshold $err_thresh --budget_limit 2000 \
                --target_valid_theta_err $valid_theta_err  --subpop_id $subpop_id > log.txt 2> logerr.txt
            if [ $? == 0 ]; then
              echo -n " ${subpop_id}"
              cp "files/data/synthetic_trn_cluster_desc.csv" $sv_fname
              mv "files/online_models/synthetic/svm/cluster/12/orig/1/subpop-${subpop_id}_online_for_real_data_tol-0.01_err-${err_thresh}.npz" \
                $atk_fname
            else
              echo ""
              echo "attack failed against subpop ${subpop_id}! exiting . . ."
              exit
            fi
          else
            echo -n " ${subpop_id}"
          fi
        done

        echo ""
        echo ""

        # move everything into a safe location
        cp "files/data/synthetic_train_test.npz" \
          "files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/synthetic_train_test.npz"
        mv "files/data/synthetic_trn_cluster_desc.csv" $dst_fname
        rm $sv_fname
      else
        echo "experiment on class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed} already complete, skipping"
      fi

    done
  done
done

# zip results together
zip -q -r "files/out/synthetic_lowerbound.zip" "files/out/synthetic_lowerbound"
