# About
This repository maintains code for the subpopulation susceptibility experiments contained in [this article](https://uvasrg.github.io/poisoning), and is based on the original model-targeted poisoning attack [repository](https://github.com/suyeecav/model-targeted-poisoning).

# Install Dependencies
The program requires the following key dependencies:
`python 2.7`, `numpy`, `cvxpy (version 0.4.11)`, `scikit-learn`, `scipy`, `matplotlib`. You can directly install these dependencies by running the following command:
```
pip install -r requirements.txt
```
Gurobi optimizer, if needed, can be setup by obtaining a license and following instructions [here](https://www.gurobi.com/documentation/9.1/quickstart_linux/software_installation_guid.html). In this case, you will also need to install the additional Python extension gurobipy using `pip install gurobipy`.

# Quick Start Guide
Please follow the instructions below to reproduce the results shown in the paper:
1. Unzip the file `files.zip` to find a folder `files`, which contains preprocessed Adult, MNIST-17 and Dogfish datasets. Synthetic subpopulations are generated programmatically and are not included in the provided datasets.
2. Susceptibility experiments from the paper are automatically run using the scripts contained in the `susceptibility` folder. Running any of the experiment scripts from the project directory will automatically carry out all the experiment steps. The scripts include the following:
- run_synthetic_viz_experiments.sh: runs subpopulation poisoning attacks against synthetic datasets. Intermediate models are saved so that the attack process can be animated. Results are saved to `files/out/synthetic_viz` and are organized by dataset parameters.
- run_adult_experiments.sh: runs subpopulation poisoning attacks against Adult dataset. Intermediate models are saved as in the synthetic experiments. Results are saved to `files/out/adult`.
- run_synthetic_lowerbound_experiments.sh: runs subpopulation poisoning attacks against synthetic datasets. Tests multiple attack strategies and target models against each subpopulation for better difficulty approximations. Results are saved to `files/out/synthetic_lowerbound`.

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
python generate_subpops.py --dataset <dataset> --subpop_type <subpop_type> --num_subpops <num_subpops>
```
where `dataset` is one of `adult` or `synthetic` and `subpop_type` is one of `cluster` or `feature`. Note that feature-based subpopulations are only supported for Adult dataset. The `num_subpops` argument controls the number of clusters when creating cluster-based subpopulations. Train set cluster identifications are written to a corresponding `labels.txt` in `files/data`, and subpopulation statistics are collected in a `desc.csv` in `files/data`. The true file names are prefixed to reflect the command line arguments.

## Generating Target Models
To generate target models for a dataset, run
```
python generate_target_theta.py --dataset <dataset> --model_type <model_type> --subpop_type <subpop_type> --valid_theta_err <valid_theta_err>
```
where `dataset`, `subpop_type`, and `model_type` are the same as used when generating subpopulations, and `valid_theta_err` is the required error rate on the target subpopulation. The optional `--save_all` flag will save multiple target models per subpopulation. Target models are saved in `files/target_classifiers`.

## Running attacks
Different experiments are implemented in different source files:
- run_kkt_online_attack.py implements the original experiments from the model-targeted attack paper. It is also used for the attacks generated in the subpopulation visualization [article](https://uvasrg.github.io/poisoning).
- run_adult_experiments.py implements the feature-matching subpopulation experiments from the article.
- run_lowerbound_experiments.py implements updated poisoning attacks which apply several target models to each subpopulation.

For detailed information about each experiment, please refer to the corresponding Python or Bash scripts.
