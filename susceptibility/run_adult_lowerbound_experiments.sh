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
subpop_fname="adult_svm_feature_selected_subpops.txt"
labels_trn_fname="adult_trn_feature_labels.txt"
labels_tst_fname="adult_tst_feature_labels.txt"

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
    python generate_subpops.py --dataset adult --lazy --subpop_type feature --subpop_ratio 0.5 \
    --tolerance 1.0 > log.txt 2> logerr.txt
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
    python generate_target_theta.py --dataset adult --model_type svm \
      --subpop_type feature --weight_decay $wdecay --save_all \
      --valid_theta_err $valid_theta_err --selection_criteria $model_selection \
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

  read -r -a subpops < "${local_dir}/${subpop_fname}"
  read -r -a counts <<< "$(tail -n 1 ${local_dir}/${subpop_fname})"
  # cat "${local_dir}/${subpop_fname}" | tail -n 1 | read -r -a counts
  # read -r -a counts <<< "$(tail -n 1 files/out/adult_lowerbound/adult_svm_feature_selected_subpops.txt)"
  # tail -n 1 | read -r -a counts < "${local_dir}/${subpop_fname}"
  echo -n "attack completed against subpops:"
  for i in "${!subpops[@]}"; do
    subpop_id=${subpops[i]}
    # count=${counts[i]}
    atk_fname="${out_dir}/subpop-${subpop_id}-atk.npz"
    if !(test -f "${atk_fname}"); then
      # override subpopulations to attack
      # echo -e "${subpop_id}\n${count}" > "${local_dir}/${subpop_fname}"

      # compute lower bounds for subpop attacks
      # python generate_subpop_lowerbounds.py --model_type svm --dataset adult \
      #   --weight_decay $wdecay --subpop_type feature --valid_theta_err $valid_theta_err \
      #   > log.txt 2> logerr.txt
      # if [ $? == 0 ]; then
      #   echo "computed lower bounds"
      # else
      #   echo "lower bound computation failed! exiting . . ."
      #   exit
      # fi

      # run attack
      python susceptibility/run_lowerbound_experiments.py --dataset adult \
          --model_type svm --subpop_type feature --weight_decay $wdecay \
          --require_acc --err_threshold $err_thresh --budget_limit 2000 \
          --target_valid_theta_err $valid_theta_err --subpop_id $subpop_id \
          --sv_im_models > log.txt 2> logerr.txt
      if [ $? == 0 ]; then
        echo -n " ${subpop_id}"
        cp "${local_dir}/${local_csv_fname}" "${out_dir}/${sv_csv_fname}"
        mv "files/online_models/adult/svm/feature/12/orig/1/subpop-${subpop_id}_online_for_real_data_tol-0.01_err-${err_thresh}.npz" \
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

  # move everything into a safe location
  mv "${local_dir}/${local_csv_fname}" "${out_dir}/${dst_csv_fname}"
  rm "${out_dir}/${sv_csv_fname}"
else
  echo "experiments on adult dataset already complete!"
fi

# zip results together
cd files/out
zip -q -r "adult_lowerbound.zip" "adult_lowerbound"
cd ../../
