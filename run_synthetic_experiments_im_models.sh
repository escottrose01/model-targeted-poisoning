#!/bin/bash

seps=($(seq 0.0 0.25 3.0))
flips=($(seq 0.0 0.1 1.0))

mkdir -p files/attack_anim

for class_sep in "${seps[@]}"; do
  for flip_y in "${flips[@]}"; do
    mkdir -p "files/attack_anim/sep${class_sep}-flip${flip_y}"
    echo "running experiment with class_sep=${class_sep}, flip_y=${flip_y}"

    # clear the stage
    rm -rf files/kkt_models \
      files/online_models \
      files/results files/target_classifiers \
      files/data/*_desc.csv \
      files/data/*_labels.txt \
      files/data/*_selected_subpops.txt

    # generate the dataset
    python generate_dataset.py --flip_y $flip_y --class_sep $class_sep --rand_seed 1 > /dev/null 2>&1
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

    python generate_target_theta.py --dataset synthetic --model_type svm \
      --subpop_type cluster --weight_decay 1e-5 --all_subpops > /dev/null 2>&1
    if [ $? == 0 ]; then
      echo "generated target theta"
    else
      echo "target theta gen failed! exiting . . ."
      exit
    fi

    python run_kkt_online_attack.py --dataset synthetic --model_type svm \
      --subpop_type cluster --weight_decay 1e-5 --require_acc --no_kkt \
      --target_model real --sv_im_models > /dev/null 2>&1
    if [ $? == 0 ]; then
      echo "completed attack!"; echo ""
    else
      echo "attack failed! exiting . . ."
      exit
    fi

    # move everything into a safe location
    mv files/online_models/synthetic/svm/cluster/12/orig/1/*.npz "files/attack_anim/sep${class_sep}-flip${flip_y}/"
    mv "files/data/synthetic_train_test.npz" "files/attack_anim/sep${class_sep}-flip${flip_y}/synthetic_train_test.npz"
  done
done
