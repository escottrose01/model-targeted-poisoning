# About
This repository maintains code for the subpopulation susceptibility experiments contained in [this article](https://uvasrg.github.io/poisoning), and is based on the original model-targeted poisoning attack [repository](https://github.com/suyeecav/model-targeted-poisoning).

# Install Dependencies
The program requires the following key dependencies:
`python 2.7`, `numpy`, `cvxpy (version 0.4.11)`, `scikit-learn`, `scipy`, `matplotlib`. You can directly install these dependencies by running the following command:
```
pip install -r requirements.txt
```
Gurobi optimizer, if needed, can be setup by obtaining a license and following instructions [here](https://www.gurobi.com/documentation/9.1/quickstart_linux/software_installation_guid.html).

# Quick Start Guide
Please follow the instructions below to reproduce the results shown in the paper:
1. Unzip the file `files.zip` to find a folder `files`, which contains preprocessed Adult, MNIST-17 and Dogfish datasets. Synthetic subpopulations are generated programmatically and are not included in the provided datasets.
2. Susceptibility experiments from the paper are automatically run using the scripts contained in the `susceptibility` folder. Running any of the experiment scripts from the project directory will automatically carry out all the experiment steps. The scripts include the following:
- run_synthetic_viz_experiments.sh: runs subpopulation poisoning attacks against synthetic datasets. Intermediate models are saved so that the attack process can be animated. Results are saved to `files/attack_anim/synthetic` and are organized by dataset parameters.
- run_adult_experiments.sh: runs subpopulation poisoning attacks against Adult dataset. Intermediate models are saved as in the synthetic experiments. Results are saved to `files/attack_anim/adult`.

Experiments can take a long time to complete (up to several days), so the scripts will skip already-completed attacks in case execution stops before all experiments are executed. More details about experiment specifications are included in the corresponding script files.

## Generating Synthetic Datasets
To generate synthetic datasets, run
```
python generate_dataset.py --flip_y <flip_y> --class_sep <class_sep> --rand_seed <seed>
```
where `flip_y` is a parameter between 0 and 1 controlling the amount of label noise, `class_sep` indicates the distance between class centers, and `seed` fixes the random seed for the dataset. The generated dataset is saved to `files/data/synthetic_train_test.npz`, and a record of the dataset parameters is written to `files/data/synthetic.json`.

## Generating Subpopulations
To generate target supopulations for a dataset, run
```
python generate_subpops.py --dataset <dataset> --subpop_type <subpop_type>
```
where `dataset` is one of `adult` or `synthetic` and `subpop_type` is one of `cluster` or `feature`. Note that feature-based subpopulations are only supported for Adult dataset.

# TODO: Document other experiment components

<!-- 1. generate_dataset.py -->
<!-- 2. generate_subpops.py -->
<!-- 3. generate_target_theta.py -->
<!-- 4. run_kkt_online_attack.py -->
<!-- 5. run_adult_experiments.py -->


<!--
# Run the Code
Please follow the instructions below to reproduce the results shown in the paper:
1. unzip the file `files.zip` and you will see folder `files`, which contains the Adult, MNIST-17 and Dogfish datasets used for evaluation in the paper. In addition, we also provide the target classifiers for each dataset in the folder `files/target_classifiers`.
2. Skip this step if you wish to use the subpopulations we provide. Else, you can generate the target subpopulations by running the command below. To generate subpopulations for other datasets, replace `adult` with `mnist_17` or `dogfish` in the command below. To use feature-based subpopulations, replace `cluster` with `feature`. You can also change the number of clusters for cluster subpopulations using `--num_clusters` or the desired subpopulation fraction for feature subpopulations using `--subpop_ratio`.
```
python generate_subpops.py --dataset adult --subpop_type cluster
```
3. Skip this step if you wish to use the target classifiers we provide. Else, you can generate the target classifiers by running the command below. To generate target classifiers for other datasets, replace `adult` with `mnist_17` or `dogfish` in the command below. To obtain results on logistic regression model, replace `svm` with `lr`. In the paper, we also improved the target model generation process for the MNIST-17 dataset and the SVM model, and if you wish to use improved target model, add `--improved` in the command below.
```
python generate_target_theta.py --dataset adult --model_type svm
```

4. To run our attack, please use the command below. Again, replace `adult` with `mnist_17` or `dogfish` to run the attack on other datasets. Replace `svm` with `lr` to run the attack on logistic regression model. For the MNIST-17 dataset, if you wish to attack the improved target classifier, add `--improved` in the command below. By feeding different values to `--rand_seed`, we can repeat the attack process for multiple times and obtain more stable results. Results in the paper can be reproduced by feeding the seeds `12`,`23`,`34`,`45` individually to `--rand_seed`.
```
python run_kkt_online_attack.py --rand_seed 12 --dataset adult --model_type svm
```

5. Once the attack is finished, run the following command to obtain the averaged results of the attack, which will be saved in directory `files/final_reslts` in `.csv` form. Replace dataset if necessary and if you used different random seeds for `--rand_seed` from above, please change the random_seeds specified in the source file. You can find the number of poisoning points used and also the computed lower bound in the `csv` file.
```
python process_avg_results.py --dataset adult --model_type svm
```

6. To generate the test accuracies (after poisoning) reported in Table 1 and Table 2 in the paper, run the following command to get the averaged results. Change datasets and model types if necessary.
```
python generate_table.py --dataset adult --model_type svm
```  

7. To reproduce the figures in the paper, run the following command. Replace the dataset if necessary and also be careful if the random seeds are different from the ones used above and change accordingly in the source file.
```
python plot_results.py --dataset adult --model_type svm
``` -->
