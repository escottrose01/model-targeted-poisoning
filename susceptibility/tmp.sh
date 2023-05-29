#!/bin/bash

# todo: add COMPAS dataset
datasets=(adult loan compas)
ratios=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
subpop_cts=(5 10 15 20 25 30 35 40 45)

# run experiments based on subpop_ratio
for subpop_ratio in "${ratios[@]}"; do
  for dataset in "${datasets[@]}"; do
    dst_fname="files/data/subpop_descs/${dataset}_trn_feature_desc_${subpop_ratio}.csv"
    dst_fname_tmp="files/data/subpop_descs/tmp/${dataset}_trn_feature_desc_${subpop_ratio}.csv"
    if (test -f "${dst_fname}") && !(test -f "${dst_fname_tmp}"); then
      echo "running ratio experiment on ${dataset} with ratio=${subpop_ratio}"

      # clear the stage
      rm -rf files/kkt_models \
        files/online_models \
        files/results files/target_classifiers \
        files/data/*_desc.csv \
        files/data/*_labels.txt \
        files/data/*_selected_subpops.txt

      python generate_subpops.py --dataset $dataset --subpop_type feature --subpop_ratio $subpop_ratio --tolerance 0.005 > /dev/null 2>&1
      python generate_target_theta.py --dataset $dataset --model_type svm --subpop_type feature --all_subpops > /dev/null 2>&1

      echo "--------------------------------------"
      echo "checking fidelity of generated supops:"
      chk_1=$(cat "files/data/${dataset}_trn_feature_desc.csv" | head -2 | tail -1 | cut -c-24)
      chk_2=$(cat $dst_fname | head -2 | tail -1 | cut -c-24)
      if [ "${chk_1}" != "${chk_2}" ]; then
        echo "WARNING: generated subpops don't seem to match, verify!"
        echo $chk_1
        echo $chk_2
        # exit
      else
        echo "generated subpops look good, continuing"
      fi
      echo "--------------------------------------"

      cp $dst_fname "files/data/${dataset}_trn_feature_desc.csv"
      python tmp.py --dataset $dataset --model_type svm --subpop_type feature --no_kkt --target_model real > /dev/null 2>&1

      # move everything into a safe location
      mv "files/data/${dataset}_trn_feature_desc.csv" $dst_fname_tmp
    else
      echo "skipping file ${dst_fname}"
    fi
  done
done

# run experiments based on subpop_cts
for subpop_ct in "${subpop_cts[@]}"; do
  for dataset in "${datasets[@]}"; do
    dst_fname="files/data/subpop_descs/${dataset}_trn_cluster_desc_${subpop_ct}.csv"
    dst_fname_tmp="files/data/subpop_descs/tmp/${dataset}_trn_cluster_desc_${subpop_ct}.csv"
    if (test -f "${dst_fname}") && !(test -f "${dst_fname_tmp}"); then
      echo "running cluster experiment on ${dataset} with subpop_ct=${subpop_ct}"

      # clear the stage
      rm -rf files/kkt_models \
        files/online_models \
        files/results files/target_classifiers \
        files/data/*_desc.csv \
        files/data/*_labels.txt \
        files/data/*_selected_subpops.txt

      python generate_subpops.py --dataset $dataset --subpop_type cluster --num_subpops $subpop_ct > /dev/null 2>&1
      python generate_target_theta.py --dataset $dataset --model_type svm --subpop_type cluster --all_subpops > /dev/null 2>&1

      echo "--------------------------------------"
      echo "checking fidelity of generated supops:"
      chk_1=$(cat "files/data/${dataset}_trn_cluster_desc.csv" | head -2 | tail -1 | cut -c-32)
      chk_2=$(cat $dst_fname | head -2 | tail -1 | cut -c-32)
      if [ "${chk_1}" != "${chk_2}" ]; then
        echo "WARNING: generated subpops don't seem to match, verify!"
        echo $chk_1
        echo $chk_2
      else
        echo "generated subpops look good, continuing"
      fi
      echo "--------------------------------------"

      cp $dst_fname "files/data/${dataset}_trn_cluster_desc.csv"
      python tmp.py --dataset $dataset --model_type svm --subpop_type cluster --no_kkt --target_model real > /dev/null 2>&1

      # move everything into a safe location
      mv "files/data/${dataset}_trn_cluster_desc.csv" $dst_fname_tmp
    else
      echo "skipping file ${dst_fname}"
    fi

    dst_fname="files/data/subpop_descs/${dataset}_trn_random_desc_${subpop_ct}.csv"
    dst_fname_tmp="files/data/subpop_descs/tmp/${dataset}_trn_random_desc_${subpop_ct}.csv"
    if (test -f "${dst_fname}") && !(test -f "${dst_fname_tmp}"); then
      echo "running random experiment on ${dataset} with subpop_ct=${subpop_ct}"

      # clear the stage
      rm -rf files/kkt_models \
        files/online_models \
        files/results files/target_classifiers \
        files/data/*_desc.csv \
        files/data/*_labels.txt \
        files/data/*_selected_subpops.txt

      python generate_subpops.py --dataset $dataset --subpop_type random --num_subpops $subpop_ct > /dev/null 2>&1
      python generate_target_theta.py --dataset $dataset --model_type svm --subpop_type random --all_subpops > /dev/null 2>&1

      echo "--------------------------------------"
      echo "checking fidelity of generated supops:"
      chk_1=$(cat "files/data/${dataset}_trn_random_desc.csv" | head -2 | tail -1 | cut -c-32)
      chk_2=$(cat $dst_fname | head -2 | tail -1 | cut -c-32)
      if [ "$chk_1" != "$chk_2" ]; then
        echo "WARNING: generated subpops don't seem to match, verify!"
        echo $chk_1
        echo $chk_2
      else
        echo "generated subpops look good, continuing"
      fi
      echo "--------------------------------------"

      cp $dst_fname "files/data/${dataset}_trn_random_desc.csv"
      python tmp.py --dataset $dataset --model_type svm --subpop_type random --no_kkt --target_model real > /dev/null 2>&1

      # move everything into a safe location
      mv "files/data/${dataset}_trn_random_desc.csv" $dst_fname_tmp
    else
      echo "skipping file ${dst_fname}"
    fi
  done
done
