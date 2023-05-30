#!/bin/bash

valid_theta_err=1.0               # target model subpopulation error requirement
err_thresh=0.5                    # attack success requirement

mkdir -p "files/subpop_descs"
mkdir -p "files/subpop_descs/semantic"
mkdir -p "files/attack_anim/"
mkdir -p "files/attack_anim/adult"

# run experiments based on subpop_ratio
dst_fname="files/subpop_descs/semantic/adult_trn_feature_desc_all.csv"
if !(test -f "${dst_fname}"); then
  # clear the stage
  rm -rf files/kkt_models \
    files/online_models \
    files/results files/target_classifiers \
    files/data/*_desc.csv \
    files/data/*_labels.txt \
    files/data/*_selected_subpops.txt

  python generate_subpops.py --dataset adult --subpop_type feature --subpop_ratio 0.5 \
  --tolerance 1.0 --lazy > /dev/null 2>&1
  if [ $? == 0 ]; then
    echo "generated subpops"
  else
    echo "subpop gen failed! exiting . . ."
    exit
  fi

  python generate_target_theta.py --dataset adult --model_type svm --subpop_type feature \
  --all_subpops --valid_theta_err $valid_theta_err > /dev/null 2>&1
  if [ $? == 0 ]; then
    echo "generated target theta"
  else
    echo "target theta gen failed! exiting . . ."
    exit
  fi

  python ./susceptibility/run_adult_experiments.py --dataset adult --model_type svm \
  --subpop_type feature --require_acc --err_threshold $err_thresh --budget_limit 16000 \
  --target_valid_theta_err $valid_theta_err --flush_freq 10 --sv_im_models > /dev/null 2>&1
  if [ $? == 0 ]; then
    echo "completed attack!"; echo ""
  else
    echo "attack failed! exiting . . ."
    exit
  fi

  mv files/online_models/adult/svm/feature/12/orig/1/*.npz \
    "files/attack_anim/adult/"

  # move everything into a safe location
  mv "files/data/adult_trn_feature_desc.csv" $dst_fname
else
  echo "file ${dst_fname} already exists, skipping"
fi
