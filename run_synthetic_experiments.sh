#!/bin/bash

seps=($(seq 0.0 0.25 3.0))
flips=($(seq 0.0 0.1 1.0))
seeds=($(seq 1 10))

mkdir -p files/subpop_descs
mkdir -p files/subpop_descs/synthetic

for class_sep in "${seps[@]}"; do
  for flip_y in "${flips[@]}"; do
    for dset_seed in "${seeds[@]}"; do
      dst_fname="files/subpop_descs/synthetic/sep${class_sep}-flip${flip_y}-seed${dset_seed}.csv"
      if !(test -f "${dst_fname}"); then
        echo "running experiment with class_sep=${class_sep}, flip_y=${flip_y}, seed=${dset_seed}"
        # clear the stage
        rm -rf files/kkt_models \
          files/online_models \
          files/results files/target_classifiers \
          files/data/*_desc.csv \
          files/data/*_labels.txt \
          files/data/*_selected_subpops.txt

        # generate the dataset
        python generate_dataset.py --flip_y $flip_y --class_sep $class_sep --rand_seed $dset_seed > /dev/null 2>&1
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
          --target_model real > /dev/null 2>&1
        if [ $? == 0 ]; then
          echo "completed attack!"; echo ""
        else
          echo "attack failed! exiting . . ."
          exit
        fi

        # move everything into a safe location
        mv "files/data/synthetic_trn_cluster_desc.csv" $dst_fname
      else
        echo "experiment on class_sep=${class_sep}, flip_y=${flip_y}, seed=${dset_seed} already complete, skipping"
      fi
    done
  done
done

# after finishing everything, combine data into single file
head -n 1 files/subpop_descs/synthetic/sep0.00-flip0.0-seed1.csv > files/subpop_descs/synthetic.csv # gets the headers
tail -n+2 -q files/subpop_descs/synthetic/*.csv >> files/subpop_descs/synthetic.csv # gets the data
