#!/bin/bash

# this script runs the lower-bound estimate poisoning experiments against adult subpopulations.

# Attack specification:
# - feature subpopulations are generated using feature matching with <= feature selections.
# - target models are generated to achieve >= 50% test error on the target subpopulation, bucketed by error rate, and selected by the criterion which minimizes loss on the non-subpopulation (collateral) clean data.

valid_theta_err=0.5               # target model subpopulation error requirement
err_thresh=0.5                    # attack success requirement
model_selection="min_collateral"  # select target model candidate which minimizes collateral damage
wdecay=0.09                       # model regularization parameter

out_dir="files/out/adult_lowerbound"
local_dir="files/data"
target_classifiers_dir="files/target_classifiers/adult/svm/feature"

dst_csv_fname="subpop_desc.csv"
local_csv_fname="adult_trn_feature_desc.csv"
sv_csv_fname="subpop_desc_sv.csv"
lowerbound_fname="lowerbounds.csv"
subpop_fname="adult_svm_feature_selected_subpops.txt"
labels_trn_fname="adult_trn_feature_labels.txt"
labels_tst_fname="adult_tst_feature_labels.txt"

num_jobs=32

trap "pkill -P $$" SIGINT SIGTERM EXIT

launch_attack() {
  local subpop_id=$1

  python susceptibility/run_lowerbound_experiments.py \
    --dataset adult \
    --model_type svm \
    --subpop_type feature \
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
    # cp "${local_dir}/${local_csv_fname}" "${out_dir}/${sv_csv_fname}"
    mv "files/online_models/adult/svm/feature/12/orig/1/subpop-${subpop_id}_online_for_real_data_tol-0.01_err-${err_thresh}.npz" \
      "${out_dir}/subpop-${subpop_id}-atk.npz"
  else
    echo "attack failed against subpop ${subpop_id}!"
  fi

  return $rv
}

mkdir -p $out_dir
mkdir -p "${out_dir}/target_classifiers"


if !(test -f "${out_dir}/${dst_csv_fname}"); then
  # clear the stage
  rm -rf files/kkt_models \
    files/online_models \
    files/results files/target_classifiers \
    files/data/*_desc.csv \
    files/data/*_labels.txt \
    files/data/*_selected_subpops.txt

  # generate or recover subpopulations
  if (test -f "${out_dir}/${sv_csv_fname}"); then
    cp "${out_dir}/${sv_csv_fname}" "${local_dir}/${local_csv_fname}"
    cp "${out_dir}/${labels_trn_fname}" "${local_dir}/${labels_trn_fname}"
    cp "${out_dir}/${labels_tst_fname}" "${local_dir}/${labels_tst_fname}"
    echo "recovered saved subpops"
  else
    python generate_subpops.py \
      --dataset adult \
      --subpop_type feature \
      --subpop_ratio 0.5 \
      --tolerance 1.0 \
      --lazy > log.txt 2> logerr.txt
    if [ $? == 0 ]; then
      cp "${local_dir}/${local_csv_fname}" "${out_dir}/${sv_csv_fname}"
      cp "${local_dir}/${labels_trn_fname}" "${out_dir}/${labels_trn_fname}"
      cp "${local_dir}/${labels_tst_fname}" "${out_dir}/${labels_tst_fname}"
      echo "generated subpops"
    else
      echo "subpop gen failed! exiting . . ."
      exit
    fi
  fi

  # generate or recover target models
  if (test -f "${out_dir}/${subpop_fname}"); then
    mkdir -p $target_classifiers_dir
    cp "${out_dir}/${subpop_fname}" "${local_dir}/${subpop_fname}"
    cp "${out_dir}/target_classifiers/"* "${target_classifiers_dir}"
    echo "recovered saved target classifiers"
  else
    python generate_target_theta.py \
      --dataset adult \
      --model_type svm \
      --subpop_type feature \
      --weight_decay $wdecay \
      --save_all \
      --valid_theta_err $valid_theta_err \
      --selection_criteria $model_selection \
      --all_subpops > log.txt 2> logerr.txt
    if [ $? == 0 ]; then
      mkdir -p "${out_dir}/target_classifiers"
      cp "${local_dir}/${local_csv_fname}" "${out_dir}/${sv_csv_fname}"
      cp "${target_classifiers_dir}/"* "${out_dir}/target_classifiers"
      cp "${local_dir}/${subpop_fname}" "${out_dir}/${subpop_fname}"
      echo "generated target theta"
    else
      echo "target theta gen failed! exiting . . ."
      exit
    fi
  fi

  # compute lower bounds for subpop attacks
  if !(test -f "${out_dir}/${lowerbound_fname}"); then
    python generate_subpop_lowerbounds.py \
      --model_type svm \
      --dataset adult \
      --weight_decay $wdecay \
      --subpop_type feature \
      --valid_theta_err $valid_theta_err > log.txt 2> logerr.txt
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

  read -r -a subpops < "${local_dir}/${subpop_fname}"
  read -r -a counts <<< "$(tail -n 1 ${local_dir}/${subpop_fname})"
  for i in "${!subpops[@]}"; do
    subpop_id=${subpops[i]}
    atk_fname="${out_dir}/subpop-${subpop_id}-atk.npz"
    if !(test -f "${atk_fname}"); then
      echo "launching attack against subpop ${subpop_id}"
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

  # move everything into a safe location
  mv "${local_dir}/${local_csv_fname}" "${out_dir}/${dst_csv_fname}"
  rm "${out_dir}/${sv_csv_fname}"
else
  echo "experiments on adult dataset already complete!"
fi

# temportary fix for min loss dif
# python susceptibility/_tmp_repair_min_loss_dif.py \
#   --dir "files/out/adult_lowerbound" \
#   --dataset adult \
#   --model_type svm \
#   --weight_decay $wdecay \
#   --subpops 4338

# combine lower bound and attack results
python susceptibility/combine_results.py "${out_dir}"

# zip results together
# cd files/out
# zip -q -r "adult_lowerbound.zip" "adult_lowerbound"
# cd ../../

