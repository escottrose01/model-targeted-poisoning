import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)

# This is needed so parallelism does not explode
parallel_procs = "2"
os.environ["OMP_NUM_THREADS"] = parallel_procs
os.environ["MKL_NUM_THREADS"] = parallel_procs
os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs

from sklearn.datasets import make_classification

import numpy as np
from sklearn import svm, linear_model
from sklearn import cluster
import csv
import pickle
import sklearn
import pandas as pd
import cvxpy as cvx

# KKT attack related modules
import kkt_attack
# from upper_bounds import hinge_loss, hinge_grad, logistic_grad
from datasets import load_dataset

import data_utils as data
import argparse
import scipy

from sklearn.externals import joblib

# import adaptive attack related functions
from utils import *
from influence_attack import influence_attack

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='test', help='file name')
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or logistic regression')
parser.add_argument('--dataset', default='adult', choices=['adult','mnist_17','2d_toy','dogfish', 'loan', 'compas', 'synthetic'])
# parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# # some params related to online algorithm, use the default
# parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss or norm')
# parser.add_argument('--incre_tol_par',default=1e-2,type=float,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,type=float,help='weight decay for regularizers')
# parser.add_argument('--err_threshold',default=None,type=float,help='target error rate')
parser.add_argument('--rand_seed',default=12,type=int,help='random seed')
# parser.add_argument('--repeat_num',default=1,type=int,help='repeat num of maximum loss diff point')
# parser.add_argument('--improved',action="store_true",help='if true, target classifier is obtained through improved process')
# parser.add_argument('--fixed_budget',default=0,type=int,help='if > 0, then run the attack for fixed number of points')
# parser.add_argument('--budget_limit', default=0, type=int, help='if > 0, then terminate attack early if iterations exceeds value')
# parser.add_argument('--require_acc',action="store_true",help='if true, terminate the algorithm when the acc requirement is achieved')
# parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature', 'random'], help='subpopulaton type: cluster or feature')
parser.add_argument('--subpops', default=1, type=int, help='number of subpopulations')
# parser.add_argument('--subpop_id', default=-1, type=int, help='if != -1, run the attack only on this subpop')
# parser.add_argument('--sv_im_models', action="store_true", help='if true, saves intermediate poisoned models')
# parser.add_argument('--target_valid_theta_err', default=None, type=float, help='classification error from target model generation')
# parser.add_argument('--flush_freq', default=-1, type=int, help='frequency with which to flush output data to file (use for large subpop counts)')

args = parser.parse_args()

dataset_name = args.dataset
# flush_freq = args.flush_freq
# subpop_type = args.subpop_type

percentile = 90
loss_percentile = 90

# if true, we generate target classifier using label flipping...
# if args.improved:
#     target_gen_proc = 'improved'
# else:
target_gen_proc = 'orig'

if dataset_name == 'mnist_17':
    args.poison_whole = True
    args.incre_tol_par = 0.1
    args.weight_decay = 0.09
    valid_theta_errs = [0.05,0.1,0.15]
elif dataset_name == 'adult':
    if args.model_type == 'lr':
        args.incre_tol_par = 0.05
    else:
        args.incre_tol_par = 0.01

    valid_theta_errs = [1.0]
elif dataset_name == '2d_toy':
    args.poison_whole = True
    if args.poison_whole:
        valid_theta_errs = [0.1,0.15]
    else:
        valid_theta_errs = [1.0]
elif dataset_name == 'dogfish':
    if args.model_type == 'lr':
        args.incre_tol_par = 1.0
    else:
        args.incre_tol_par = 2.0

    args.poison_whole = True
    args.weight_decay = 1.1

    if args.poison_whole:
        valid_theta_errs = [0.1,0.2,0.3]
    else:
        valid_theta_errs = [0.9]
elif dataset_name in ['loan', 'compas', 'synthetic']:
    if args.model_type == 'lr':
        args.incre_tol_par = 0.05
    else:
        args.incre_tol_par = 0.01

    valid_theta_errs = [1.0]

# if args.err_threshold is not None:
#     valid_theta_errs = [args.err_threshold]

if args.model_type == 'svm':
    ScikitModel = svm_model
    model_grad = hinge_grad
else:
    ScikitModel = logistic_model
    model_grad = logistic_grad

learning_rate = 0.01
######################################################################

################# Main body of work ###################
# create files that store clustering info
# make_dirs(args)

# load data
X_train, y_train, X_test, y_test = load_dataset(args.dataset)

if min(y_test)>-1:
    y_test = 2*y_test-1
if min(y_train) > -1:
    y_train = 2*y_train - 1

full_x = np.concatenate((X_train,X_test),axis=0)
full_y = np.concatenate((y_train,y_test),axis=0)
if args.dataset == "2d_toy":
    # get the min and max value of features
    x_pos_min, x_pos_max = np.amin(full_x[full_y == 1]),np.amax(full_x[full_y == 1])
    x_neg_min, x_neg_max = np.amin(full_x[full_y == -1]),np.amax(full_x[full_y == -1])
    x_pos_tuple = (x_pos_min,x_pos_max)
    x_neg_tuple = (x_neg_min,x_neg_max)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
elif args.dataset in ["adult","mnist_17","loan", 'compas', 'synthetic']:
    x_pos_tuple = (0,1)
    x_neg_tuple = (0,1)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
elif args.dataset == "dogfish":
    x_pos_min, x_pos_max = np.amin(full_x,axis=0),np.amax(full_x,axis=0)
    x_neg_min, x_neg_max = np.amin(full_x,axis=0),np.amax(full_x,axis=0)
    x_pos_tuple = (x_pos_min,x_pos_max)
    x_neg_tuple = (x_neg_min,x_neg_max)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
else:
    x_pos_tuple = None
    x_neg_tuple = None

# data preprocessers for the current data
class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
X_train,
y_train,
percentile=percentile)

if dataset_name == "adult":
    pois_rates = [0.05]
elif dataset_name == "mnist_17":
    pois_rates = [0.2]

# if args.subpop_id != -1:
#     subpop_inds = [args.subpop_id]
subpop_inds = list(range(args.subpops))

# train unpoisoned model
C = 1.0 / (X_train.shape[0] * args.weight_decay)
fit_intercept = True
model = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter=32000)
model.fit(X_train, y_train)

# report performance of clean model
clean_acc = model.score(X_test,y_test)
margins = y_train*(X_train.dot(model.coef_.reshape(-1)) + model.intercept_)
clean_loss, _ = calculate_loss(margins)
clean_loss += (args.weight_decay/2) * (np.linalg.norm(model.coef_)**2 + model.intercept_**2)

params = np.reshape(model.coef_, -1)
bias = model.intercept_[0]

X_train_cp, y_train_cp = np.copy(X_train), np.copy(y_train)

# start the complete process
# ATK_NPZ_FORMAT = 'subpop-{0}-atk.npz'
for kk, subpop_ind in enumerate(subpop_inds):
    try:
        fname = os.path.join(args.dir, 'subpop-{}-atk.npz'.format(subpop_ind))
        attack_results = np.load(fname)

        attack_log = attack_results['attack_log']
        attack_stats = attack_results['attack_stats'].item()
        trn_sbcl = attack_results['trn_sbcl']
        tst_sbcl = attack_results['tst_sbcl']
        trn_non_sbcl = attack_results['trn_non_sbcl']
        tst_non_sbcl = attack_results['tst_non_sbcl']

        # compute the best loss difference
        min_loss_dif = float('inf')
        lu_lb = float('inf')
        for atk in attack_log:
            atk['loss_diff'] = float('inf')
            if atk['attack_tag'] in ['mtp-1', 'influence']:
                theta_atk, bias_atk = atk['theta_atk'], atk['bias_atk']
                margins = y_train*(X_train.dot(theta_atk) + bias_atk)
                train_loss, _ = calculate_loss(margins)
                train_loss += (args.weight_decay / 2.0) * (np.linalg.norm(theta_atk)**2 + bias_atk**2)
                atk["loss_diff"] = train_loss - clean_loss

            if 'theta_p' in atk.keys():
                theta_p, bias_p = atk['theta_p'], atk['bias_p']
                margins = y_train*(X_train.dot(theta_p) + bias_p)
                train_loss, _ = calculate_loss(margins)
                train_loss += (args.weight_decay / 2.0) * (np.linalg.norm(theta_p)**2 + bias_p**2)
                atk["loss_diff"] = min(train_loss - clean_loss, atk["loss_diff"])
            
            min_loss_dif = min(min_loss_dif, atk["loss_diff"])
            if "loss_dif" in atk.keys():
                del atk["loss_dif"] # ugh.. typo

        # print('Subpop {} min loss dif: {}'.format(fname, min_loss_dif))
        print('lu_lb: {}; other_lb: {}'.format(lu_lb, attack_stats['Min Lower Bound']))


        attack_stats['Min Loss Diff'] = min_loss_dif
        if 'Lu Lower Bound' in attack_stats.keys():
            del attack_stats['Lu Lower Bound']
        if "Min Loss Dif" in attack_stats.keys():
            del attack_stats["Min Loss Dif"]

        np.savez(
            fname,
            attack_log=attack_log,
            attack_stats=attack_stats,
            trn_sbcl=trn_sbcl,
            tst_sbcl=tst_sbcl,
            trn_non_sbcl=trn_non_sbcl,
            tst_non_sbcl=tst_non_sbcl,
            allow_pickle=True
        )

    except IOError:
        print('File not found for {}'.format(fname))