#!/bin/bash

# this script runs the lower-bound estimate poisoning experiments against synthetic subpopulations.

# Dataset specification:
# - datasets are generated over a 13x11 grid of (class separation, label noise) dataset parameter pairs.
# - for each parameter combination, 10 different random seeds are used (seeds are reused between parameter combinations).

# Attack specification:
# - 16 cluster subpopulations are generated for each dataset.
# - target models are generated to achieve >= 50% test error on the target subpopulation, bucketed by error rate, and selected by the criterion which minimizes loss on the non-subpopulation (collateral) clean data.

valid_theta_err=0.5               # target model subpopulation error requirement
err_thresh=0.5                    # attack success requirement
model_selection="min_collateral"  # select target model candidate which minimizes collateral damage
wdecay=5e-4                       # model regularization parameter

out_dir="files/out/synthetic_lowerbound"
local_dir="files/data"
target_classifiers_dir="files/target_classifiers/synthetic/svm/cluster"

dst_csv_fname="subpop_desc.csv"
local_csv_fname="synthetic_trn_cluster_desc.csv"
sv_csv_fname="subpop_desc_sv.csv"
lowerbound_fname="lowerbounds.csv"
subpop_fname="synthetic_svm_cluster_selected_subpops.txt"
labels_trn_fname="synthetic_trn_cluster_labels.txt"
labels_tst_fname="synthetic_tst_cluster_labels.txt"

num_jobs=16

trap "pkill -P $$" SIGINT SIGTERM EXIT
set -e

launch_attack() {
  local subpop_id=$1

  python susceptibility/run_lowerbound_experiments.py \
    --dataset synthetic \
    --model_type svm \
    --subpop_type cluster \
    --weight_decay $wdecay \
    --err_threshold $err_thresh \
    --budget_limit 2000 \
    --target_valid_theta_err $valid_theta_err \
    --subpop_id $subpop_id \
    --require_acc \
    --sv_im_models > log.txt 2> logerr.txt

  rv=$?

  if [ $rv == 0 ]; then
    echo "attack completed against subpop ${subpop_id}"
    mv "files/online_models/synthetic/svm/cluster/12/orig/1/subpop-${subpop_id}_online_for_real_data_tol-0.01_err-${err_thresh}.npz" \
      "${out_dir}/subpop-${subpop_id}-atk.npz"
    # mv "files/online_models/synthetic/svm/cluster/12/orig/1/subpop-${subpop_id}_online_for_real_data_tol-0.01_err-${err_thresh}.npz" \
    #           $atk_fname
  else
    echo "attack failed against subpop ${subpop_id}!"
  fi

  return $rv
}

seps=($(seq 0.0 0.25 3.00))        # class separation dataset parameter
flips=($(seq 0.0 0.1 1.0))        # label noise dataset parameter
seeds=($(seq 1 10))               # dataset seed

mkdir -p "files/out/synthetic_lowerbound"
for seed in "${seeds[@]}"; do
  for class_sep in "${seps[@]}"; do
    for flip_y in "${flips[@]}"; do
      out_dir="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}"
      
      mkdir -p $out_dir
      mkdir -p "${out_dir}/target_classifiers"


      # mkdir -p "files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}"
      # dst_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/subpop_desc.csv"
      # sv_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/subpop_desc_tmp.csv"
      # lowerbound_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/lowerbounds.csv"
      # subpop_fname="files/data/synthetic_svm_cluster_selected_subpops.txt"

      if !(test -f "${out_dir}/${dst_csv_fname}"); then
        echo "running experiment with class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed}"

        # clear the stage
        rm -rf files/kkt_models \
          files/online_models \
          files/results files/target_classifiers \
          files/data/*_desc.csv \
          files/data/*_labels.txt \
          files/data/*_selected_subpops.txt

        # generate the dataset
        python generate_dataset.py \
          --flip_y $flip_y \
          --class_sep $class_sep \
          --rand_seed $seed > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "generated dataset"
        else
          echo "dataset gen failed! exiting . . ."
          exit
        fi

        # generate subpopulations
        python generate_subpops.py \
          --dataset synthetic \
          --subpop_type cluster \
          --num_subpops 16 \
          --lazy > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "generated subpops"
        else
          echo "subpop gen failed! exiting . . ."
          exit
        fi

        # generate target models
        python generate_target_theta.py \
          --dataset synthetic \
          --model_type svm \
          --subpop_type cluster \
          --weight_decay $wdecay \
          --valid_theta_err $valid_theta_err \
          --selection_criteria $model_selection \
          --save_all \
          --all_subpops > log.txt 2> logerr.txt
        if [ $? == 0 ]; then
          echo "generated target theta"
        else
          echo "target theta gen failed! exiting . . ."
          exit
        fi

        # compute lower bounds for subpop attacks
        if !(test -f "${out_dir}/{$lowerbound_fname}"); then
          python generate_subpop_lowerbounds.py \
            --model_type svm \
            --dataset synthetic \
            --weight_decay $wdecay \
            --subpop_type cluster \
            --valid_theta_err $valid_theta_err \
            > log.txt 2> logerr.txt
          if [ $? == 0 ]; then
            mv "${local_dir}/${lowerbound_fname}" "${out_dir}/${lowerbound_fname}"
            echo "computed lower bounds"
          else
            echo "lower bound computation failed! exiting . . ."
            exit
          fi
        else
          echo "recovered saved lower bounds"
        fi

        # recover preexisting .csv data, if exists
        # no longer meaningful
        # if (test -f "${out_dir}/${sv_csv_fname}"); then
        #   cp "${out_dir}/${$sv_csv_fname}" "files/data/synthetic_trn_cluster_desc.csv"
        # fi

        # iterate over all subpops (individually)
        read -r -a subpops < "${local_dir}/${subpop_fname}"
        for i in "${!subpops[@]}"; do
          subpop_id=${subpops[$i]}
          atk_fname="${out_dir}/subpop-${subpop_id}-atk.npz"
          if !(test -f "${atk_fname}"); then
            echo "running attack against subpop ${subpop_id}"
            launch_attack $subpop_id &
            
            while [[ $(jobs -r -p | wc -l) -ge $num_jobs ]]; do
              wait -n
              if [ $? != 0 ]; then
                echo "attack failed; bailing . . ."
                exit
              fi
            done
          else
            echo "Skipping subpop ${subpop_id} (already attacked)"
          fi
        done

        wait

        # read -r -a subpops < $subpop_fname
        # echo -n "completed attacks against subpops:"
        # for subpop_id in "${subpops[@]}"; do
        #   # check if attack is already completed against this subpop
        #   atk_fname="files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/subpop-${subpop_id}-atk.npz"
        #   if !(test -f "${atk_fname}"); then
        #     # run attack
        #     python susceptibility/run_lowerbound_experiments.py --dataset synthetic \
        #         --model_type svm --subpop_type cluster --weight_decay $wdecay \
        #         --require_acc --err_threshold $err_thresh --budget_limit 2000 \
        #         --target_valid_theta_err $valid_theta_err --subpop_id $subpop_id \
        #         --sv_im_models > log.txt 2> logerr.txt
        #     if [ $? == 0 ]; then
        #       echo -n " ${subpop_id}"
        #       cp "files/data/synthetic_trn_cluster_desc.csv" $sv_fname
        #       mv "files/online_models/synthetic/svm/cluster/12/orig/1/subpop-${subpop_id}_online_for_real_data_tol-0.01_err-${err_thresh}.npz" \
        #         $atk_fname
        #     else
        #       echo ""
        #       echo "attack failed against subpop ${subpop_id}! exiting . . ."
        #       exit
        #     fi
        #   else
        #     echo -n " ${subpop_id}"
        #   fi
        # done

        echo ""
        echo ""

        # move everything into a safe location
        mv "${local_dir}/${local_csv_fname}" "${out_dir}/${dst_csv_fname}"
        cp "${local_dir}/synthetic_train_test.npz" "${out_dir}/synthetic_train_test.npz"
        rm "${out_dir}/${sv_csv_fname}"

        # cp "files/data/synthetic_train_test.npz" \
        #   "files/out/synthetic_lowerbound/sep${class_sep}-flip${flip_y}-seed${seed}/synthetic_train_test.npz"
        # mv "files/data/synthetic_trn_cluster_desc.csv" $dst_fname
        # rm $sv_fname
      else
        echo "experiment on class_sep=${class_sep}, flip_y=${flip_y}, seed=${seed} already complete, skipping"
      fi

      # Combine lower bound and attack results
      # python susceptibility/combine_results.py "${out_dir}"

    done
  done
done

# concat csv files
python susceptibility/concat_csvs.py "files/out/synthetic_lowerbound"

# # zip results together
# zip -q -r "files/out/synthetic_lowerbound.zip" "files/out/synthetic_lowerbound"
