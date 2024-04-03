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
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or logistic regression')
parser.add_argument('--dataset', default='adult', choices=['adult','mnist_17','2d_toy','dogfish', 'loan', 'compas', 'synthetic'])
parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# some params related to online algorithm, use the default
parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss or norm')
parser.add_argument('--incre_tol_par',default=1e-2,type=float,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,type=float,help='weight decay for regularizers')
parser.add_argument('--err_threshold',default=None,type=float,help='target error rate')
parser.add_argument('--rand_seed',default=12,type=int,help='random seed')
parser.add_argument('--repeat_num',default=1,type=int,help='repeat num of maximum loss diff point')
parser.add_argument('--improved',action="store_true",help='if true, target classifier is obtained through improved process')
parser.add_argument('--fixed_budget',default=0,type=int,help='if > 0, then run the attack for fixed number of points')
parser.add_argument('--budget_limit', default=0, type=int, help='if > 0, then terminate attack early if iterations exceeds value')
parser.add_argument('--require_acc',action="store_true",help='if true, terminate the algorithm when the acc requirement is achieved')
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature', 'random'], help='subpopulaton type: cluster or feature')
parser.add_argument('--subpop_id', default=-1, type=int, help='if != -1, run the attack only on this subpop')
parser.add_argument('--sv_im_models', action="store_true", help='if true, saves intermediate poisoned models')
parser.add_argument('--target_valid_theta_err', default=None, type=float, help='classification error from target model generation')
parser.add_argument('--flush_freq', default=-1, type=int, help='frequency with which to flush output data to file (use for large subpop counts)')

args = parser.parse_args()

dataset_name = args.dataset
flush_freq = args.flush_freq
subpop_type = args.subpop_type

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

if args.err_threshold is not None:
    valid_theta_errs = [args.err_threshold]

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
make_dirs(args)

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

# do clustering and test if these fit previous clusters
if args.poison_whole:
    subpop_inds, subpop_cts = [0], [0]
else:
    trn_subpop_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
    if not os.path.isfile(trn_subpop_fname):
        print("please first generate the target classifier and obtain subpop info!")
        sys.exit(1)

    with open(trn_subpop_fname, 'r') as f:
        trn_all_subpops = [np.array(map(int, line.split())) for line in f]
    tst_subpop_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
    with open(tst_subpop_fname, 'r') as f:
        tst_all_subpops = [np.array(map(int, line.split())) for line in f]

    # find the selected clusters and corresponding subpop size
    cls_fname = 'files/data/{}_{}_{}_selected_subpops.txt'.format(dataset_name, args.model_type, subpop_type)
    selected_subpops = np.loadtxt(cls_fname)
    subpop_inds = selected_subpops[0]
    subpop_cts = selected_subpops[1]

if dataset_name == "adult":
    pois_rates = [0.05]
elif dataset_name == "mnist_17":
    pois_rates = [0.2]

if args.subpop_id != -1:
    if args.subpop_id not in subpop_inds:
        print('Could not find target subpopulation in selected subpops!')
        sys.exit(1)
    subpop_inds = [args.subpop_id]
    subpop_cts = subpop_cts[subpop_inds.index(args.subpop_id)]

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
for valid_theta_err in valid_theta_errs:
    args.err_threshold = valid_theta_err
    if args.target_valid_theta_err is None:
        target_valid_theta_err = valid_theta_err
    else:
        target_valid_theta_err = args.target_valid_theta_err

    # for kk in range(len(subpop_inds)):
    for kk, subpop_ind in enumerate(subpop_inds):
        # subpop_ind = int(subpop_inds[kk])
        if args.poison_whole:
            tst_sub_x, tst_sub_y = X_test, y_test
            tst_nsub_x, tst_nsub_y = X_test,y_test
            trn_sub_x, trn_sub_y = X_train, y_train
            trn_nsub_x, trn_nsub_y = X_train, y_train
        else:
            tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
            trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
            # indices of points belong to subpop
            tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, y_test, y_train)

            # get the corresponding points in the dataset
            tst_sub_x, tst_sub_y = X_test[tst_sbcl], y_test[tst_sbcl]
            tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], y_test[tst_non_sbcl]
            trn_sub_x, trn_sub_y = X_train_cp[trn_sbcl], y_train_cp[trn_sbcl]
            trn_nsub_x, trn_nsub_y = X_train_cp[trn_non_sbcl], y_train_cp[trn_non_sbcl]

            if dataset_name in ['adult', 'loan', 'compas']:
                assert (tst_sub_y == -1).all()
                assert (trn_sub_y == -1).all()
            else:
                mode_tst = scipy.stats.mode(tst_sub_y)
                mode_trn = scipy.stats.mode(trn_sub_y)
                major_lab_trn = mode_trn.mode[0]
                major_lab_tst = mode_tst.mode[0]
                assert (tst_sub_y == major_lab_tst).all()
                assert (trn_sub_y == major_lab_tst).all()

        subpop_data = [trn_sub_x,trn_sub_y,trn_nsub_x,trn_nsub_y,\
            tst_sub_x,tst_sub_y,tst_nsub_x,tst_nsub_y]

        test_target = model.score(tst_sub_x, tst_sub_y)
        test_collat = model.score(tst_nsub_x, tst_nsub_y)
        train_target = model.score(trn_sub_x, trn_sub_y)
        train_all = model.score(X_train, y_train)
        train_target_acc = model.score(trn_sub_x[trn_sub_y == -1],  trn_sub_y[trn_sub_y == -1])
        train_subpop_acc = model.score(X_train[trn_subpop_inds], y_train[trn_subpop_inds])

        fname = open('files/target_classifiers/{}/{}/{}/orig_thetas_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type,subpop_ind,target_valid_theta_err), 'rb')
        f = pickle.load(fname)

        # Compute clean model performance
        clean_all_margins = model.decision_function(X_train)
        clean_subpop_margins = model.decision_function(X_train[trn_subpop_inds])
        clean_target_margins = model.decision_function(trn_sub_x)

        clean_all_loss, _ = calculate_loss(y_train * clean_all_margins)
        clean_subpop_loss, _ = calculate_loss(y_train[trn_subpop_inds] * clean_subpop_margins)
        clean_target_loss, _ = calculate_loss(trn_sub_y * clean_target_margins)

        thetas = f['thetas']
        biases = f['biases']

        attack_log = []

        # step 1: perform attacks using label-flip target models. Save lower bounds and attack info
        for target_theta, target_bias in zip(thetas, biases):
            poisons_all = {}
            poisons_all["X_poison"] = X_train
            poisons_all["Y_poison"] = y_train

            if not fit_intercept:
                target_bias = 0

            ## apply online learning algorithm to provide lower bound and candidate attack ##
            C = 1.0 / (X_train.shape[0] * args.weight_decay)
            curr_model = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter = 32000)
            curr_model.fit(X_train, y_train)

            target_model = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter = 32000)
            target_model.fit(X_train, y_train)

            # default setting for target model is the actual model
            target_model.coef_= np.array([target_theta])
            target_model.intercept_ = np.array([target_bias])

            online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
            best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses, ol_tol_params, \
            max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs, im_models = incre_online_learning(X_train,
                                                                                    y_train,
                                                                                    X_test,
                                                                                    y_test,
                                                                                    curr_model,
                                                                                    target_model,
                                                                                    x_lim_tuples,
                                                                                    args,
                                                                                    ScikitModel,
                                                                                    target_model_type = "real",
                                                                                    attack_num_poison = 0,
                                                                                    kkt_tol_par = None,
                                                                                    subpop_data = subpop_data,
                                                                                    target_poisons = poisons_all)
            # retrain the online model based on poisons from our adaptive attack
            if len(online_poisons_y) > 0:
                online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                online_poisons_y = np.concatenate(online_poisons_y,axis=0)
                online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
            else:
                online_poisons_x = np.array(online_poisons_x)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = X_train
                online_full_y = y_train

            # retrain the model based poisons from online learning
            C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
            fit_intercept = True
            model_p_online = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=args.rand_seed,
                verbose=False,
                max_iter = 32000)
            model_p_online.fit(online_full_x, online_full_y)

            attack_log.append(dict(
                X_psn=online_poisons_x,
                y_psn=online_poisons_y,
                theta_p=target_theta,
                bias_p=target_bias,
                theta_atk=model_p_online.coef_.reshape((-1,)),
                bias_atk=model_p_online.intercept_.item(),
                best_lower_bound=best_lower_bound,
                im_models=im_models,
                attack_tag='mtp-1'
            ))

        # step 2: perform online attack using induced models as target models. Save lower bounds and attack info.
        # since the only purpose of this step is to produce a more optimal attack, we set a more restrictive budget.
        args.budget_limit = min([len(v['y_psn']) for v in attack_log])
        for atk in attack_log[:]:
            target_theta = atk['theta_atk']
            target_bias = atk['bias_atk']

            poisons_all = {}
            poisons_all["X_poison"] = X_train
            poisons_all["Y_poison"] = y_train

            if not fit_intercept:
                target_bias = 0

            ## apply online learning algorithm to provide lower bound and candidate attack ##
            C = 1.0 / (X_train.shape[0] * args.weight_decay)
            curr_model = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter = 32000)
            curr_model.fit(X_train, y_train)

            target_model = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter = 32000)
            target_model.fit(X_train, y_train)

            target_model.coef_= np.array([target_theta])
            target_model.intercept_ = np.array([target_bias])

            online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
            best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses, ol_tol_params, \
            max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs, im_models = incre_online_learning(X_train,
                                                                                    y_train,
                                                                                    X_test,
                                                                                    y_test,
                                                                                    curr_model,
                                                                                    target_model,
                                                                                    x_lim_tuples,
                                                                                    args,
                                                                                    ScikitModel,
                                                                                    target_model_type = "real",
                                                                                    attack_num_poison = 0,
                                                                                    kkt_tol_par = None,
                                                                                    subpop_data = subpop_data,
                                                                                    target_poisons = poisons_all)
            # retrain the online model based on poisons from our adaptive attack
            if len(online_poisons_y) > 0:
                online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                online_poisons_y = np.concatenate(online_poisons_y,axis=0)
                online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
            else:
                online_poisons_x = np.array(online_poisons_x)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = X_train
                online_full_y = y_train

            # retrain the model based poisons from online learning
            C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
            fit_intercept = True
            model_p_online = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=args.rand_seed,
                verbose=False,
                max_iter = 32000)
            model_p_online.fit(online_full_x, online_full_y)

            attack_log.append(dict(
                X_psn=online_poisons_x,
                y_psn=online_poisons_y,
                theta_p=target_theta,
                bias_p=target_bias,
                theta_atk=model_p_online.coef_.reshape((-1,)),
                bias_atk=model_p_online.intercept_.item(),
                best_lower_bound=best_lower_bound,
                im_models=im_models,
                attack_tag='mtp-2'
            ))

        assert len(attack_log) == 2*len(thetas), 'failed to record all attacks so far for MTP'

        # step 3: perform kkt attacks using label-flip target models. Save attack info (but do not generate lower bounds)
        for atk in attack_log[:]:
            target_theta = atk['theta_p']
            target_bias = atk['bias_p']
            lower_bound = atk['best_lower_bound']
            best_atk_poisons = len(atk['y_psn'])

            # attack parameters
            percentile = 90
            loss_percentile = 90
            use_slab = False
            use_loss = False
            use_l2 = False
            epsilon_increment = 0.005
            sub_frac = 1

            if lower_bound == best_atk_poisons or best_atk_poisons == 0:
                continue

            # some dummy models for various purposes
            model_dumb = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter=32000)
            model_dumb1 = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter=32000)
            kkt_model_p = ScikitModel(
                        C=C,
                        tol=1e-8,
                        fit_intercept=fit_intercept,
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter=32000)
            model_dumb.fit(X_train, y_train)
            model_dumb1.fit(X_train[0:2000], y_train[0:2000])
            kkt_model_p.fit(X_train, y_train)

            # use lower bounds as basis for kkt fractions
            kkt_fraction_res = 4
            min_frac = float(lower_bound) / float(X_train.shape[0])
            max_frac = float(best_atk_poisons) / float(X_train.shape[0])
            kkt_fractions = np.linspace(min_frac, max_frac, kkt_fraction_res, endpoint=False)

            kkt_fraction_max_loss_diffs = []
            kkt_fraction_norm_diffs = []
            kkt_fraction_acc_scores = []
            kkt_fraction_num_poisons = []
            kkt_fraction_loss_on_clean = []
            # setup the kkt attack class
            two_class_kkt, clean_grad_at_target_theta, target_bias_grad, max_losses = kkt_attack.kkt_setup(
                target_theta,
                target_bias,
                X_train_cp, y_train_cp,
                X_test, y_test,
                dataset_name,
                percentile,
                loss_percentile,
                model_dumb,
                model_grad,
                class_map,
                use_slab,
                use_loss,
                use_l2,
                x_pos_tuple=x_pos_tuple,
                x_neg_tuple=x_neg_tuple,
                model_type=args.model_type)


            for kkt_fraction in kkt_fractions:
                # eps pairs and objective for choosing best kkt classifier
                epsilon_pairs = []
                best_grad_diff_norm = 1e10

                kkt_num_points = int(len(online_poisons_y)*kkt_fraction)
                kkt_fraction_num_poisons.append(kkt_num_points)
                total_epsilon = float(kkt_num_points)/X_train.shape[0]

                target_grad = clean_grad_at_target_theta + ((1 + total_epsilon) * args.weight_decay * target_theta)
                if args.model_type == 'svm':
                    # Note: this inequality is based on eps_pos + eps_neg = eps; grad_w = sum(y_i) = 0 (further split y between pos and negative classes);
                    # only in the case of svm, we can derive the exact form and for logistic regression, the expression cannot be directly computed
                    # specifically, the epsilon is related to the prediction of poisoned points, which is the optimization variable we want to obtain
                    epsilon_neg = (total_epsilon - target_bias_grad) / 2
                    epsilon_pos = total_epsilon - epsilon_neg

                    if (epsilon_neg >= 0) and (epsilon_neg <= total_epsilon):
                        epsilon_pairs.append((epsilon_pos, epsilon_neg))

                for epsilon_pos in np.arange(0, total_epsilon + 1e-6, epsilon_increment):
                    epsilon_neg = total_epsilon - epsilon_pos
                    epsilon_pairs.append((epsilon_pos, epsilon_neg))

                for epsilon_pos, epsilon_neg in epsilon_pairs:
                    try:
                        if args.model_type == 'svm':
                            X_modified, Y_modified, obj, x_pos, x, num_pos, num_neg = kkt_attack.kkt_attack(
                                two_class_kkt,
                                target_grad, target_theta,
                                total_epsilon * sub_frac, epsilon_pos * sub_frac, epsilon_neg * sub_frac,
                                X_train_cp, y_train_cp,
                                class_map, centroids, centroid_vec, sphere_radii, slab_radii,
                                target_bias, target_bias_grad, max_losses)
                        elif args.model_type == 'lr':
                            # newly implemeted logistic regression solver based on gradient descend strategies
                            lr = 0.1
                            num_steps = 20000
                            trials = 10
                            X_modified, Y_modified, obj, x_pos, x, num_pos, num_neg = kkt_attack.kkt_for_lr(
                                X_train.shape[1],args,
                                target_grad,target_theta, target_bias,
                                total_epsilon * sub_frac, epsilon_pos * sub_frac, epsilon_neg * sub_frac,
                                X_train_cp, y_train_cp,
                                x_pos_tuple = x_pos_tuple,x_neg_tuple = x_neg_tuple,
                                lr=lr,num_steps=num_steps,trials=trials)

                        # separate out the poisoned points
                        idx_poison = slice(X_train.shape[0], X_modified.shape[0])
                        idx_clean = slice(0, X_train.shape[0])

                        X_poison = X_modified[idx_poison,:]
                        Y_poison = Y_modified[idx_poison]

                        # retrain the model
                        C = 1.0 / (X_modified.shape[0] * args.weight_decay)
                        model_p = ScikitModel(
                            C=C,
                            tol=1e-8,
                            fit_intercept=fit_intercept,
                            random_state=args.rand_seed,
                            verbose=False,
                            max_iter = 32000)
                        model_p.fit(X_modified, Y_modified)

                        # acc on subpop and rest of pops
                        trn_total_acc = model_p.score(X_train, y_train)
                        trn_target_acc = model_p.score(trn_sub_x, trn_sub_y)
                        trn_collat_acc = model_p.score(trn_nsub_x, trn_nsub_y)
                        tst_total_acc = model_p.score(X_test, y_test)
                        tst_target_acc = model_p.score(tst_sub_x, tst_sub_y)
                        tst_collat_acc = model_p.score(tst_nsub_x, tst_nsub_y)

                        if tst_target_acc <= valid_theta_err:
                            # only add these data if met attack objective
                            attack_log.append(dict(
                                X_psn=X_poison,
                                y_psn=Y_poison,
                                theta_p=target_theta,
                                bias_p=target_bias,
                                theta_atk=model_p.coef_.reshape((-1,)),
                                bias_atk=model_p.intercept_.item(),
                                attack_tag='kkt'
                            ))
                            # all_poisons_x.append(X_poison)
                            # all_poisons_y.append(Y_poison)
                            # all_theta_p.append(target_theta)
                            # all_bias_p.append(target_bias)
                            # all_theta_atk.append(model_p_online.coef_.reshape((-1,)))
                            # all_bias_atk.append(model_p_online.intercept_.item())
                            # attack_tags.append('kkt')
                            break # further attacks at this epsilon will not improve estimates
                    except cvx.error.SolverError:
                        pass
                        # Gurobi can fail to find points, in this case ignore

        # step 4: perform influence attack. Binary search on the number of points needed.
        lo, hi = 0, max([len(v['y_psn']) for v in attack_log])
        num_steps = 200
        X_best, y_best, history_best = None, None, None
        while hi - lo > 1:
            # launch attack
            mid = (lo + hi) // 2
            X_psn, y_psn, history = influence_attack(
                X_train,
                y_train,
                X_test,
                y_test,
                x_lim_tuples,
                subpop_data,
                args,
                ScikitModel=ScikitModel,
                num_poisons=mid,
                num_steps=num_steps,
                lr=5e-2
            )

            X_cur = np.concatenate([X_train, X_psn], axis=0)
            y_cur = np.concatenate([y_train, y_psn], axis=0)
            C = 1.0 / (X_cur.shape[0] * args.weight_decay)
            score = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=True,
                random_state=args.rand_seed,
                verbose=False,
                max_iter=32000
            ).fit(X_cur, y_cur).score(tst_sub_x, tst_sub_y)

            if score <= 0.5:
                hi = mid
                X_best, y_best, history_best = X_psn, y_psn, history
            else:
                lo = mid

        X_psn, y_psn, history = X_best, y_best, history_best
        if X_psn is not None:
            # retrain the model based poisons from influence attack
            X_full = np.concatenate((X_train, X_psn), axis=0)
            y_full = np.concatenate((y_train, y_psn), axis=0)
            C = 1.0 / (X_full.shape[0] * args.weight_decay)
            fit_intercept = True
            model_p_online = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=args.rand_seed,
                verbose=False,
                max_iter = 32000)
            model_p_online.fit(X_full, y_full)

        # save results, if attack was successful
        if X_psn is not None:
            attack_log.append(dict(
                X_psn=X_psn,
                y_psn=y_psn,
                theta_atk=model_p_online.coef_.reshape((-1,)),
                bias_atk=model_p_online.intercept_.item(),
                attack_tag='influence'
            ))
            

        filename = 'files/online_models/{}/{}/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type,args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)

        # compute the best attack
        best_ix = np.argmin([len(v['y_psn']) for v in attack_log])
        proj_constraint_size = max(
            proj_constraint_size(model.coef_, x_lim_tuples[0]),
            proj_constraint_size(model.coef_, x_lim_tuples[1])
        )
        proj_sep, proj_std = proj_separability(model.coef_, X_train, y_train)
        lower_bounds = [v['best_lower_bound'] for v in attack_log if 'best_lower_bound' in v.keys()]

        # compute the best loss difference
        min_loss_diff = float('inf')
        for atk in attack_log:
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

        attack_stats = {
            'Clean Overall Acc': train_all,
            'Clean Subpop Acc': train_subpop_acc,
            'Clean Target Acc': train_target_acc,
            'Clean Overall Loss': clean_all_loss,
            'Clean Subpop Loss': clean_subpop_loss,
            'Clean Target Loss': clean_target_loss,
            'Clean Overall Avg Margin': np.mean(clean_all_margins),
            'Clean Subpop Avg Margin': np.mean(clean_subpop_margins),
            'Clean Target Avg Margin': np.mean(clean_target_margins),
            'Min Lower Bound': min(lower_bounds),
            'Best Attack Poisons': min([len(v['y_psn']) for v in attack_log]),
            'Attempts': len(attack_log),
            'Best Strategy': attack_log[best_ix]['attack_tag'],
            'Projected Constraint Size': proj_constraint_size,
            'Projected Separability': proj_sep,
            'Projected Std': proj_std,
            'Min Loss Diff': min_loss_diff
        }

        np.savez(filename,
                attack_log=attack_log,
                attack_stats=attack_stats,
                trn_sbcl = trn_sbcl,
                tst_sbcl = tst_sbcl,
                trn_non_sbcl = trn_non_sbcl,
                tst_non_sbcl = tst_non_sbcl,
                allow_pickle=True
            )
