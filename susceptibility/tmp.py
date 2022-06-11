from sklearn.datasets import make_classification
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, linear_model
from sklearn import cluster
import csv
import pickle
import sklearn
from sklearn.metrics import hinge_loss
import pandas as pd

# import cvxpy as cvx

# KKT attack related modules
import kkt_attack
# from upper_bounds import hinge_loss, hinge_grad, logistic_grad
from datasets import load_dataset

import data_utils as data
import argparse
import os
import sys
import scipy

from sklearn.externals import joblib

# import adaptive attack related functions
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or rlogistic regression')
# ol: target classifier is from the adapttive attack, kkt: target is from kkt attack, real: actual classifier, compare: compare performance
# of kkt attack and adaptive attack using same stop criteria
parser.add_argument('--target_model', default='all',help='set your target classifier, options: kkt, ol, real, compare, all')
parser.add_argument('--dataset', default='adult', choices=['adult','mnist_17','2d_toy','dogfish', 'loan', 'compas'])
parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# some params related to online algorithm, use the default
parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss or norm')
parser.add_argument('--incre_tol_par',default=1e-2,type=float,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,type=float,help='weight decay for regularizers')
parser.add_argument('--err_threshold',default=1.0,type=float,help='target error rate')
parser.add_argument('--rand_seed',default=12,type=int,help='random seed')
parser.add_argument('--repeat_num',default=1,type=int,help='repeat num of maximum loss diff point')
parser.add_argument('--improved',action="store_true",help='if true, target classifier is obtained through improved process')
parser.add_argument('--fixed_budget',default=0,type=int,help='if > 0, then run the attack for fixed number of points')
parser.add_argument('--require_acc',action="store_true",help='if true, terminate the algorithm when the acc requirement is achieved')
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature', 'random'], help='subpopulaton type: cluster or feature')
parser.add_argument('--no_kkt', action="store_true", help='if true, does not run the kkt attack')

args = parser.parse_args()

####################### set up the poisoning attack parameters #######################################

# KKT attack specific parameters
percentile = 90
loss_percentile = 90
use_slab = False
use_loss = False
use_l2 = False
dataset_name = args.dataset
# if true, we generate target classifier using label flipping...
if args.improved:
    target_gen_proc = 'improved'
else:
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
elif dataset_name in ['loan', 'compas']:
    if args.model_type == 'lr':
        args.incre_tol_par = 0.05
    else:
        args.incre_tol_par = 0.01

    valid_theta_errs = [1.0]

if args.model_type == 'svm':
    print("chosen model: svm")
    ScikitModel = svm_model
    model_grad = hinge_grad
else:
    print("chosen model: lr")
    ScikitModel = logistic_model
    model_grad = logistic_grad

learning_rate = 0.01

dataset_name = args.dataset
subpop_type = args.subpop_type
######################################################################

################# Main body of work ###################
# creat files that store clustering info
make_dirs(args)

# load data
X_train, y_train, X_test, y_test = load_dataset(args.dataset)
print("shape of train and test data:")
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

print("max feature values:",np.amin(X_train),np.amax(X_train))


if min(y_test)>-1:
    y_test = 2*y_test-1
if min(y_train) > -1:
    y_train = 2*y_train - 1

full_x = np.concatenate((X_train,X_test),axis=0)
full_y = np.concatenate((y_train,y_test),axis=0)
if args.dataset == "2d_toy":
    # get the min and max value of features
    # if constrained:
    x_pos_min, x_pos_max = np.amin(full_x[full_y == 1]),np.amax(full_x[full_y == 1])
    x_neg_min, x_neg_max = np.amin(full_x[full_y == -1]),np.amax(full_x[full_y == -1])
    x_pos_tuple = (x_pos_min,x_pos_max)
    x_neg_tuple = (x_neg_min,x_neg_max)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
    print("max values of the features of the chosen dataset:")
    print(x_pos_min,x_pos_max,x_neg_min,x_neg_max)
elif args.dataset in ["adult","mnist_17","loan", 'compas']:
    x_pos_tuple = (0,1)
    x_neg_tuple = (0,1)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
elif args.dataset == "dogfish":
    x_pos_min, x_pos_max = np.amin(full_x,axis=0),np.amax(full_x,axis=0)
    x_neg_min, x_neg_max = np.amin(full_x,axis=0),np.amax(full_x,axis=0)
    x_pos_tuple = (x_pos_min,x_pos_max)
    x_neg_tuple = (x_neg_min,x_neg_max)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
    print("max values of the features of the chosen dataset:")
    print(x_pos_min.shape,x_pos_max.shape,x_neg_min.shape,x_neg_max.shape)
    print(np.amin(x_pos_min),np.amax(x_pos_max),np.amin(x_neg_min),np.amax(x_neg_max))
else:
    x_pos_tuple = None
    x_neg_tuple = None

# data preprocessers for the current data
class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
X_train,
y_train,
percentile=percentile)

# do clustering and test if these fit previous clusters
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

# search step size for kkt attack
epsilon_increment = 0.005
# train unpoisoned model
C = 1.0 / (X_train.shape[0] * args.weight_decay)
fit_intercept = True
model = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter=1000)
model.fit(X_train, y_train)

# some models defined only to use as an instance of classification model
model_dumb = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter=1000)
model_dumb.fit(X_train, y_train)

model_dumb1 = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter=1000)
model_dumb1.fit(X_train[0:2000], y_train[0:2000])

# will be used as the model generated from the KKT attack
kkt_model_p = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter=1000)
kkt_model_p.fit(X_train, y_train)

# report performance of clean model
clean_acc = model.score(X_test,y_test)
print("Clean Total Acc:",clean_acc)
margins = y_train*(X_train.dot(model.coef_.reshape(-1)) + model.intercept_)
clean_total_loss = np.sum(np.maximum(1-margins, 0))
print("clean model loss on train:",clean_total_loss)
# print("clean model theta and bias:",model.coef_,model.intercept_)

params = np.reshape(model.coef_, -1)
bias = model.intercept_[0]

X_train_cp, y_train_cp = np.copy(X_train), np.copy(y_train)

# start the complete process
for valid_theta_err in valid_theta_errs:
    print("Attack Target Classifiers with Expected Error Rate:",valid_theta_err)
    args.err_threshold = valid_theta_err
    # open the files to write key info
    if args.target_model == "all":
        kkt_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/kkt_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        kkt_lower_bound_writer = csv.writer(kkt_lower_bound_file, delimiter=str(' '))

        real_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/real_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        real_lower_bound_writer = csv.writer(real_lower_bound_file, delimiter=str(' '))

        ol_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/ol_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        ol_lower_bound_writer = csv.writer(ol_lower_bound_file, delimiter=str(' '))

        # for subpop descriptions
        trn_desc_fname = 'files/data/{}_trn_{}_desc.csv'.format(dataset_name, subpop_type)
        trn_df = pd.read_csv(trn_desc_fname)
    elif args.target_model == "kkt":
        kkt_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/kkt_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        kkt_lower_bound_writer = csv.writer(kkt_lower_bound_file, delimiter=str(' '))
    elif args.target_model == "real":
        real_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/real_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        real_lower_bound_writer = csv.writer(real_lower_bound_file, delimiter=str(' '))

        # for subpop descriptions
        trn_desc_fname = 'files/data/{}_trn_{}_desc.csv'.format(dataset_name, subpop_type)
        trn_df = pd.read_csv(trn_desc_fname)
    elif args.target_model == "ol":
        ol_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/ol_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        ol_lower_bound_writer = csv.writer(ol_lower_bound_file, delimiter=str(' '))
    elif args.target_model == "compare":
        compare_lower_bound_file = open('files/results/{}/{}/{}/{}/{}/{}/compare_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        compare_lower_bound_writer = csv.writer(compare_lower_bound_file, delimiter=str(' '))

    for kk in range(len(subpop_inds)):
        subpop_ind = int(subpop_inds[kk])
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

            # make sure subpop is from class -1
            # assert (tst_sub_y == -1).all()
            # assert (trn_sub_y == -1).all()
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

        print(tst_nsub_x.shape, tst_nsub_y.shape, subpop_ind)
        test_target = model.score(tst_sub_x, tst_sub_y)
        test_collat = model.score(tst_nsub_x, tst_nsub_y)
        train_target = model.score(trn_sub_x, trn_sub_y)
        train_all = model.score(X_train, y_train)
        train_target_acc = model.score(trn_sub_x[trn_sub_y == -1],  trn_sub_y[trn_sub_y == -1])
        train_subpop_acc = model.score(X_train[trn_subpop_inds], y_train[trn_subpop_inds])

        print("----------Subpop Indx: {}------".format(subpop_ind))
        print('Clean Overall Test Acc : %.3f' % model.score(X_test, y_test))
        print('Clean Test Target Acc : %.3f' % test_target)
        print('Clean Test Collat Acc : %.3f' % test_collat)
        print('Clean Overall Train Acc : %.3f' % model.score(X_train, y_train))
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print("shape of subpopulations",trn_sub_x.shape,trn_nsub_x.shape,tst_sub_x.shape,tst_nsub_x.shape)

        orig_model_acc_scores = []
        orig_model_acc_scores.append(model.score(X_test, y_test))
        orig_model_acc_scores.append(test_target)
        orig_model_acc_scores.append(test_collat)
        orig_model_acc_scores.append(model.score(X_train, y_train))
        orig_model_acc_scores.append(model.score(trn_sub_x, trn_sub_y))
        orig_model_acc_scores.append(model.score(trn_nsub_x,trn_nsub_y))

        # load target classifiers
        if not args.improved:
            if args.poison_whole:
                fname = open('files/target_classifiers/{}/{}/{}/orig_best_theta_whole_err-{}'.format(dataset_name,args.model_type, subpop_type, valid_theta_err), 'rb')

            else:
                fname = open('files/target_classifiers/{}/{}/{}/orig_best_theta_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type, subpop_ind,valid_theta_err), 'rb')
        else:
            if args.poison_whole:
                fname = open('files/target_classifiers/{}/{}/{}/improved_best_theta_whole_err-{}'.format(dataset_name,args.model_type, subpop_type,valid_theta_err), 'rb')

            else:
                fname = open('files/target_classifiers/{}/{}/{}/improved_best_theta_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type,subpop_ind,valid_theta_err), 'rb')
        f = pickle.load(fname)
        best_target_theta = f['thetas']
        best_target_bias = f['biases']

        margins = trn_sub_y * (trn_sub_x.dot(best_target_theta) + best_target_bias)
        pois_target_loss, _ = calculate_loss(margins)
        margins = trn_sub_y * (trn_sub_x.dot(model.coef_.flatten()) + model.intercept_)
        clean_target_loss, _ = calculate_loss(margins)

        margins = y_train[trn_subpop_inds] * (X_train[trn_subpop_inds].dot(best_target_theta) + best_target_bias)
        pois_subpop_loss, _ = calculate_loss(margins)
        margins = y_train[trn_subpop_inds] * (X_train[trn_subpop_inds].dot(model.coef_.flatten()) + model.intercept_)
        clean_subpop_loss, _ = calculate_loss(margins)

        margins = y_train * (X_train.dot(best_target_theta) + best_target_bias)
        pois_all_loss, _ = calculate_loss(margins)
        margins = y_train * (X_train.dot(model.coef_.flatten()) + model.intercept_)
        clean_all_loss, _ = calculate_loss(margins)

        cosine_sim = best_target_theta.dot(model.coef_.flatten()) / (np.linalg.norm(best_target_theta) * np.linalg.norm(model.coef_.flatten()))
        attack_euc_distance = np.linalg.norm(best_target_theta - model.coef_.flatten())

        print('Clean Target LOSS: %.5f' % clean_target_loss)
        print('Pois Target LOSS: %.5f' % pois_target_loss)
        print('Clean Subpop LOSS: %.5f' % clean_subpop_loss)
        print('Pois Subpop LOSS: %.5f' % pois_subpop_loss)
        print('Clean All LOSS: %.5f' % clean_all_loss)
        print('Pois All LOSS: %.5f' % pois_all_loss)
        print('Cosine Sim: %.5f' % cosine_sim)
        print('Model distance: %.5f' % attack_euc_distance)

        # l2_norm = np.sqrt(np.linalg.norm(best_target_theta)**2 + best_target_bias**2)
        # print("norm of the target model weights:",l2_norm)
        # sys.exit(0)
        sub_frac = 1

        # fname = 'files/target_classifiers/{}/orig_best_poison_subpop_{}_err-{}.npz'.format(dataset_name,args.model_type,int(subpop_ind),valid_theta_err)
        # poisons_all = np.load(fname)
        # X_Poison = poisons_all["X_poison"]
        # Y_Poison = poisons_all["Y_poison"]
        # no longer need it, so just fill some random stuff
        poisons_all = {}
        poisons_all["X_poison"] = X_train
        poisons_all["Y_poison"] = y_train

        # # print info of the target classifier # #
        print("--- Acc Info of Actual Target Classifier ---")
        target_model_acc_scores = []
        margins = tst_sub_y*(tst_sub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_target_err = calculate_loss(margins)
        margins =tst_nsub_y*(tst_nsub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_collat_err = calculate_loss(margins)
        margins =y_test*(X_test.dot(best_target_theta) + best_target_bias)
        _, ideal_total_err = calculate_loss(margins)
        generalizability = ideal_target_err
        print("Ideal Total Test Acc:",1-ideal_total_err)
        print("Ideal Target Test Acc:",1-ideal_target_err)
        print("Ideal Collat Test Acc:",1-ideal_collat_err)
        target_model_acc_scores.append(1-ideal_total_err)
        target_model_acc_scores.append(1-ideal_target_err)
        target_model_acc_scores.append(1-ideal_collat_err)

        margins = trn_sub_y*(trn_sub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_target_err = calculate_loss(margins)
        margins =trn_nsub_y*(trn_nsub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_collat_err = calculate_loss(margins)
        margins =y_train*(X_train.dot(best_target_theta) + best_target_bias)
        _, ideal_total_err = calculate_loss(margins)
        print("Ideal Total Train Acc:",1-ideal_total_err)
        print("Ideal Target Train Acc:",1-ideal_target_err)
        print("Ideal Collat Train Acc:",1-ideal_collat_err)
        target_model_acc_scores.append(1-ideal_total_err)
        target_model_acc_scores.append(1-ideal_target_err)
        target_model_acc_scores.append(1-ideal_collat_err)

        # # just to make sure one subpop of 2d toy example will terminate
        # if 1-ideal_target_err > 1-args.err_threshold:
        #     print("the target classifier does not satisfy the attack goal, skip the rest!")
        #     continue
        # store the lower bound and actual poisoned points
        kkt_target_lower_bound_and_attacks = []
        ol_target_lower_bound_and_attacks = []
        real_target_lower_bound_and_attacks = []
        compare_target_lower_bound_and_attacks = []

        print("************** Target Classifier for Subpop:{} ***************".format(subpop_ind))
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
                    max_iter = 1000)
        curr_model.fit(X_train, y_train)

        target_model = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=args.rand_seed,
                    verbose=False,
                    max_iter = 1000)
        target_model.fit(X_train, y_train)

        collat_model = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter=1000
        )
        collat_model.fit(trn_nsub_x, trn_nsub_y)

        avg_importance = np.linalg.norm(collat_model.coef_ - curr_model.coef_) / trn_sub_x.shape[0]

        # default setting for target model is the actual model
        target_model.coef_= np.array([best_target_theta])
        target_model.intercept_ = np.array([best_target_bias])
        ##### Start the evaluation of different target classifiers #########
        # if args.target_model == "real" or args.target_model == "all":
        #     print("------- Use Actual Target model as Target Model -----")
        #     if args.poison_whole:
        #         filename = 'files/online_models/{}/{}/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #     else:
        #         filename = 'files/online_models/{}/{}/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type,args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #     if not os.path.isfile(filename):
        #         # start the evaluation process
        #         print("[Sanity Real] Acc of current model:",curr_model.score(X_test,y_test),curr_model.score(X_train,y_train))
        #
        #         online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
        #         best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses,\
        #         ol_tol_params, max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs = incre_online_learning(X_train,
        #                                                                                 y_train,
        #                                                                                 X_test,
        #                                                                                 y_test,
        #                                                                                 curr_model,
        #                                                                                 target_model,
        #                                                                                 x_lim_tuples,
        #                                                                                 args,
        #                                                                                 ScikitModel,
        #                                                                                 target_model_type = "real",
        #                                                                                 attack_num_poison = 0,
        #                                                                                 kkt_tol_par = None,
        #                                                                                 subpop_data = subpop_data,
        #                                                                                 target_poisons = poisons_all)
        #         # retrain the online model based on poisons from our adaptive attack
        #         if len(online_poisons_y) > 0:
        #             print("Original Shape of online x and online y:",np.array(online_poisons_x).shape,np.array(online_poisons_y).shape)
        #             online_poisons_x = np.concatenate(online_poisons_x,axis=0)
        #             online_poisons_y = np.concatenate(online_poisons_y,axis=0)
        #             online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
        #             online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
        #         else:
        #             print("online learning does not make progress and using original model!")
        #             online_poisons_x = np.array(online_poisons_x)
        #             online_poisons_y = np.array(online_poisons_y)
        #             online_full_x = X_train
        #             online_full_y = y_train
        #         print("shape of online poisoned points:",online_poisons_x.shape,online_poisons_y.shape)
        #         print("shape of full poisoned points:",online_full_x.shape,online_full_y.shape)
        #         # retrain the model based poisons from online learning
        #         C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
        #         fit_intercept = True
        #         model_p_online = ScikitModel(
        #             C=C,
        #             tol=1e-8,
        #             fit_intercept=fit_intercept,
        #             random_state=args.rand_seed,
        #             verbose=False,
        #             max_iter = 1000)
        #         model_p_online.fit(online_full_x, online_full_y)
        #
        #         # save the data and model for producing the online models
        #         if args.poison_whole:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/whole-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         else:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/subpop-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         joblib.dump(model_p_online, filename)
        #
        #         if args.poison_whole:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         else:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type,args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         np.savez(filename,
        #                 online_poisons_x = online_poisons_x,
        #                 online_poisons_y = online_poisons_y,
        #                 best_lower_bound = best_lower_bound,
        #                 conser_lower_bound = conser_lower_bound,
        #                 best_max_loss_x = best_max_loss_x,
        #                 best_max_loss_y = best_max_loss_y,
        #                 target_poison_max_losses = target_poison_max_losses,
        #                 current_total_losses = current_total_losses,
        #                 max_loss_diffs = max_loss_diffs_reg,
        #                 lower_bounds = lower_bounds,
        #                 ol_tol_params = ol_tol_params,
        #                 online_acc_scores = np.array(online_acc_scores),
        #                 norm_diffs = norm_diffs
        #                 )
        #     else:
        #         if args.poison_whole:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         else:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type, subpop_type,args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         data_info = np.load(filename)
        #         online_poisons_x = data_info["online_poisons_x"]
        #         online_poisons_y = data_info["online_poisons_y"]
        #         best_lower_bound = data_info["best_lower_bound"]
        #         conser_lower_bound = data_info["conser_lower_bound"]
        #         best_max_loss_x = data_info["best_max_loss_x"]
        #         best_max_loss_y = data_info["best_max_loss_y"]
        #         target_poison_max_losses = data_info["target_poison_max_losses"]
        #         current_total_losses = data_info["current_total_losses"]
        #         max_loss_diffs_reg = data_info["max_loss_diffs"]
        #         lower_bounds = data_info["lower_bounds"]
        #         ol_tol_params = data_info["ol_tol_params"]
        #         ol_tol_par = ol_tol_params[-1]
        #         online_acc_scores = data_info["online_acc_scores"]
        #         norm_diffs = data_info['norm_diffs']
        #         if args.poison_whole:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/whole-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.model_type,subpop_type,args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         else:
        #             filename = 'files/online_models/{}/{}/{}/{}/{}/{}/subpop-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.model_type,subpop_type, args.rand_seed,target_gen_proc,args.repeat_num,subpop_ind,args.incre_tol_par,valid_theta_err)
        #         model_p_online = joblib.load(open(filename, 'rb'))

            # write to subpop description info
            # print('subpop {} used {} poisons, subpop acc = {:.3f}, target acc = {:.3f}'.format(subpop_ind, len(online_poisons_y), train_subpop_acc, train_target_acc))
            # print('poisoned model accuracy on subpop: {:.3f}'.format(model_p_online.score(trn_sub_x, trn_sub_y)))
            # trn_df.loc[subpop_ind, 'Clean Overall Acc'] = train_all
            # trn_df.loc[subpop_ind, 'Clean Subpop Acc'] = train_subpop_acc
            # trn_df.loc[subpop_ind, 'Clean Target Acc'] = train_target_acc
            # trn_df.loc[subpop_ind, 'Num Poisons'] = len(online_poisons_y)

        trn_df.loc[subpop_ind, 'Clean Target Loss'] = clean_target_loss
        trn_df.loc[subpop_ind, 'Poisoned Target Loss'] = pois_target_loss
        trn_df.loc[subpop_ind, 'Clean Subpop Loss'] = clean_subpop_loss
        trn_df.loc[subpop_ind, 'Poisoned Subpop Loss'] = pois_subpop_loss
        trn_df.loc[subpop_ind, 'Clean Overall Loss'] = clean_all_loss
        trn_df.loc[subpop_ind, 'Poisoned Overall Loss'] = pois_all_loss
        trn_df.loc[subpop_ind, 'Model Cosine Similarity'] = cosine_sim
        trn_df.loc[subpop_ind, 'Model Eucl. Distance'] = attack_euc_distance
        trn_df.loc[subpop_ind, 'Generalizability'] = generalizability
        trn_df.loc[subpop_ind, 'Avg Importance'] = avg_importance
        pos = trn_df.loc[subpop_ind, 'Num Positive'] / trn_df.loc[subpop_ind, 'Subpop Pts']
        trn_df.loc[subpop_ind, 'Gini Impurity'] = 1 - (pos**2 + (1-pos)**2)

trn_df.to_csv(trn_desc_fname, index=False)
