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

# import cvxpy as cvx

# KKT attack related modules
import kkt_attack
# from upper_bounds import hinge_loss, hinge_grad, logistic_grad
from datasets import load_dataset

import data_utils as data
import argparse
import os
import sys

from sklearn.externals import joblib

# import adaptive attack related functions
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',default='svm',help='victim model type: SVM or rlogistic regression')
# ol: target classifier is from the adapttive attack, kkt: target is from kkt attack, real: actual classifier, compare: compare performance
# of kkt attack and adaptive attack using same stop criteria
parser.add_argument('--target_model', default='all',help='set your target classifier, options: kkt, ol, real, compare, all')
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, synthetic")
parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# some params related to online algorithm, use the default
parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss or norm')
parser.add_argument('--incre_tol_par',default=1e-2,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,help='weight decay for regularizers')

args = parser.parse_args()

####################### set up the poisoning attack parameters #######################################

# KKT attack specific parameters
percentile = 90
loss_percentile = 90
use_slab = False
use_loss = False
use_l2 = False
dataset_name = args.dataset

assert dataset_name in ['adult','mnist_17']
if dataset_name == 'mnist_17':
    args.poison_whole = True
    # see if decreasing by half helps
    # args.incre_tol_par = 0.05
    args.weight_decay = 0.09
elif dataset_name == 'adult':
    args.weight_decay = 1e-5


if args.model_type == 'svm':
    print("chosen model: svm")
    ScikitModel = svm_model
    model_grad = hinge_grad
else:
    print("chosen model: lr")
    ScikitModel = logistic_model
    model_grad = logistic_grad

# norm_sq_constraint = 1.0
max_loss_tol_par = 1e-2

learning_rate = 0.01
online_alg = "incremental"

######################################################################

################# Main body of work ###################
# creat files that store clustering info
make_dirs(args.dataset)
if args.target_model == "all":
    kkt_lower_bound_file = open('files/results/{}/kkt_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    kkt_lower_bound_writer = csv.writer(kkt_lower_bound_file, delimiter=str(' ')) 

    real_lower_bound_file = open('files/results/{}/real_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    real_lower_bound_writer = csv.writer(real_lower_bound_file, delimiter=str(' ')) 

    ol_lower_bound_file = open('files/results/{}/ol_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    ol_lower_bound_writer = csv.writer(ol_lower_bound_file, delimiter=str(' ')) 
elif args.target_model == "kkt":
    kkt_lower_bound_file = open('files/results/{}/kkt_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    kkt_lower_bound_writer = csv.writer(kkt_lower_bound_file, delimiter=str(' ')) 
elif args.target_model == "real":
    real_lower_bound_file = open('files/results/{}/real_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    real_lower_bound_writer = csv.writer(real_lower_bound_file, delimiter=str(' ')) 
elif args.target_model == "ol":
    ol_lower_bound_file = open('files/results/{}/ol_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    ol_lower_bound_writer = csv.writer(ol_lower_bound_file, delimiter=str(' ')) 
elif args.target_model == "compare":
    compare_lower_bound_file = open('files/results/{}/compare_lower_bound_and_attacks_tol-{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
    compare_lower_bound_writer = csv.writer(compare_lower_bound_file, delimiter=str(' ')) 

# load data
X_train, y_train, X_test, y_test = load_dataset(args.dataset)

if min(y_test)>-1:
    y_test = 2*y_test-1
if min(y_train) > -1:
    y_train = 2*y_train - 1

full_x = np.concatenate((X_train,X_test),axis=0)
full_y = np.concatenate((y_train,y_test),axis=0)
if args.dataset == "synthetic":
    # get the min and max value of features
    # if constrained:
    x_pos_min, x_pos_max = np.amin(full_x[full_y == 1]),np.amax(full_x[full_y == 1])
    x_neg_min, x_neg_max = np.amin(full_x[full_y == -1]),np.amax(full_x[full_y == -1])
    x_pos_tuple = (x_pos_min,x_pos_max)
    x_neg_tuple = (x_neg_min,x_neg_max)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
    print("max values of the features of synthetic dataset:")
    print(x_pos_min,x_pos_max,x_neg_min,x_neg_max)
elif args.dataset in ["adult","mnist_17"]:
    x_pos_tuple = (0,1)
    x_neg_tuple = (0,1)
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
# do clustering and test if these fit previous clusters
if args.poison_whole:
    cl_inds, cl_cts = [0], [0]
else:
    cls_fname = 'files/data/{}_trn_cluster_labels.txt'.format(args.dataset)
    if os.path.isfile(cls_fname):
        trn_km = np.loadtxt(cls_fname)
        cls_fname = 'files/data/{}_tst_cluster_labels.txt'.format(args.dataset)
        tst_km = np.loadtxt(cls_fname)
    else:
        print("please first generate the target classifier and obtain subpop info!")
        sys.exit(1) 
    # find the selected clusters and corresponding subpop size
    # cl_inds, cl_cts = np.unique(trn_km, return_counts=True)
    cls_fname = 'files/data/{}_selected_subpops.txt'.format(dataset_name)
    selected_subpops = np.loadtxt(cls_fname)
    cl_inds = selected_subpops[0]
    cl_cts = selected_subpops[1]

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
            random_state=24,
            verbose=False,
            max_iter = 1000)
model.fit(X_train, y_train)

# use this to check ideal classifier performance for KKT attack
model_dumb = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
model_dumb.fit(X_train, y_train)

model_dumb1 = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
model_dumb1.fit(X_train[0:2000], y_train[0:2000])

kkt_model_p = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
kkt_model_p.fit(X_train, y_train)

# report performance of clean model
clean_acc = model.score(X_test,y_test)
print("Clean Total Acc:",clean_acc)
margins = y_train*(X_train.dot(model.coef_.reshape(-1)) + model.intercept_)
clean_total_loss = np.sum(np.maximum(1-margins, 0))
print("clean model loss on train:",clean_total_loss)
# print("clean model theta and bias:",model.coef_,model.intercept_)

X_train_cp, y_train_cp = np.copy(X_train), np.copy(y_train)

for kk in range(len(cl_inds)):
    cl_ind = int(cl_inds[kk])
    if args.poison_whole:
        tst_sub_x, tst_sub_y = X_test, y_test 
        tst_nsub_x, tst_nsub_y = X_test,y_test
        trn_sub_x, trn_sub_y = X_train, y_train
        trn_nsub_x, trn_nsub_y = X_train, y_train
    else:
        tst_sbcl = np.where(np.logical_and(tst_km==cl_ind,y_test == -1))
        trn_sbcl = np.where(np.logical_and(trn_km==cl_ind,y_train == -1))
        tst_non_sbcl = np.where(np.logical_or(tst_km!=cl_ind,y_test != -1))
        trn_non_sbcl = np.where(np.logical_or(trn_km!=cl_ind,y_train != -1))
        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y = X_train_cp[trn_sbcl], y_train_cp[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train_cp[trn_non_sbcl], y_train_cp[trn_non_sbcl]
    
        # make sure subpop is from class -1
        assert (tst_sub_y == -1).all()
        assert (trn_sub_y == -1).all()

    subpop_data = [trn_sub_x,trn_sub_y,trn_nsub_x,trn_nsub_y,\
        tst_sub_x,tst_sub_y,tst_nsub_x,tst_nsub_y]

    test_target = model.score(tst_sub_x, tst_sub_y)
    test_collat = model.score(tst_nsub_x, tst_nsub_y)

    print("----------Subpop Indx: {}------".format(cl_ind))
    print('Clean Overall Test Acc : %.3f' % model.score(X_test, y_test))
    print('Clean Test Target Acc : %.3f' % test_target)
    print('Clean Test Collat Acc : %.3f' % test_collat)
    print('Clean Overall Train Acc : %.3f' % model.score(X_train, y_train))
    print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
    print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))

    orig_model_acc_scores = []
    orig_model_acc_scores.append(model.score(X_test, y_test))
    orig_model_acc_scores.append(test_target)
    orig_model_acc_scores.append(test_collat)
    orig_model_acc_scores.append(model.score(X_train, y_train))
    orig_model_acc_scores.append(model.score(trn_sub_x, trn_sub_y))
    orig_model_acc_scores.append(model.score(trn_nsub_x,trn_nsub_y))

    # load target classifiers
    if args.poison_whole:
        fname = open('files/target_classifiers/{}/best_theta_whole'.format(dataset_name), 'rb')  
        
    else:
        fname = open('files/target_classifiers/{}/best_theta_subpop_{}'.format(dataset_name,cl_ind), 'rb')  
    f = pickle.load(fname)
    best_target_theta = f['thetas']
    best_target_bias = f['biases']
    sub_frac = 1

    # # print info of the target classifier # #
    print("--- Acc Info of Actual Target Classifier ---")
    target_model_acc_scores = []
    margins = tst_sub_y*(tst_sub_x.dot(best_target_theta) + best_target_bias)
    _, ideal_target_err = calculate_loss(margins)
    margins =tst_nsub_y*(tst_nsub_x.dot(best_target_theta) + best_target_bias)
    _, ideal_collat_err = calculate_loss(margins)
    margins =y_test*(X_test.dot(best_target_theta) + best_target_bias)
    _, ideal_total_err = calculate_loss(margins)
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

    # store the lower bound and actual poisoned points
    kkt_target_lower_bound_and_attacks = []
    ol_target_lower_bound_and_attacks = []
    real_target_lower_bound_and_attacks = []
    compare_target_lower_bound_and_attacks = []

    # for total_epsilon in pois_rates:
    epsilon_pairs = []
    best_target_acc = 1e10
    # if args.poison_whole:
    #     filename = 'files/kkt_models/{}/whole-{}_model.sav'.format(dataset_name,cl_ind)
    # else:
    #     filename = 'files/kkt_models/{}/subpop-{}_model.sav'.format(dataset_name,cl_ind)
    # if not os.path.isfile(filename):
    print("************** Target Classifier for Subpop:{} ***************".format(cl_ind))
    if not fit_intercept:
        target_bias = 0

    ## apply online learning algorithm to provide lower bound and candidate attack ##
    C = 1.0 / (X_train.shape[0] * args.weight_decay)
    curr_model = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
    curr_model.fit(X_train, y_train)

    target_model = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
    target_model.fit(X_train, y_train)
    # default setting for target model is the actual model
    target_model.coef_= np.array([best_target_theta])
    target_model.intercept_ = np.array([best_target_bias])
    ##### Start the evaluation of different target classifiers #########
    if args.target_model == "real" or args.target_model == "all":
        print("------- Use Actual Target model as Target Model -----")
        if args.poison_whole:
            filename = 'files/online_models/{}/whole-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/online_models/{}/subpop-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        if not os.path.isfile(filename):
            # start the evaluation process
            online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
            best_max_loss_y, ol_tol_par, target_poison_max_losses, total_loss_diffs,\
            ol_tol_params, max_loss_diffs_reg, lower_bounds = incre_online_learning(X_train,
                                                                                    y_train,
                                                                                    curr_model,
                                                                                    target_model,
                                                                                    x_lim_tuples,
                                                                                    args,
                                                                                    ScikitModel,
                                                                                    target_model_type = "real",
                                                                                    attack_num_poison = 0,
                                                                                    kkt_tol_par = None)
            # retrain the online model based on poisons from our adaptive attack
            if len(online_poisons_y) > 0:
                online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
            else:
                print("online learning does not make progress and using original model!")
                online_poisons_x = np.array(online_poisons_x)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = X_train
                online_full_y = y_train
            print("shape of online poisoned points:",online_poisons_x.shape,online_poisons_y.shape)
            print("shape of full poisoned points:",online_full_x.shape,online_full_y.shape)
            # retrain the model based poisons from online learning
            C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
            fit_intercept = True
            model_p_online = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
            model_p_online.fit(online_full_x, online_full_y) 

            # save the data and model for producing the online models
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            joblib.dump(model_p_online, filename)

            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            np.savez(filename,
                    online_poisons_x = online_poisons_x,
                    online_poisons_y = online_poisons_y,
                    best_lower_bound = best_lower_bound,
                    conser_lower_bound = conser_lower_bound,
                    best_max_loss_x = best_max_loss_x,
                    best_max_loss_y = best_max_loss_y,
                    target_poison_max_losses = target_poison_max_losses,
                    total_loss_diffs = total_loss_diffs, 
                    max_loss_diffs = max_loss_diffs_reg,
                    lower_bounds = lower_bounds,
                    ol_tol_params = ol_tol_params
                    )

            ## draw the trend and also save some important statistics ##
            plt.clf()
            if args.poison_whole:
                filename = 'files/results/{}/whole-{}_max_loss_diff_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/results/{}/subpop-{}_max_loss_diff_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            plt.plot(np.arange(len(max_loss_diffs_reg)), np.squeeze(np.array(max_loss_diffs_reg)), 'r--')
            plt.savefig(filename)   

            plt.clf()
            if args.poison_whole:
                filename = 'files/results/{}/whole-{}_total_loss_diff_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/results/{}/subpop-{}_total_loss_diff_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)

            plt.plot(np.arange(len(total_loss_diffs)), np.squeeze(np.array(total_loss_diffs)), 'r--')
            plt.savefig(filename)  

            # plot the curve of lower bound and loss value w.r.t. iterations, also save these values for future use
            plt.clf()
            if args.poison_whole:
                filename = 'files/results/{}/whole-{}_lower_bound_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/results/{}/subpop-{}_lower_bound_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            plt.plot(np.arange(len(lower_bounds)), np.squeeze(np.array(lower_bounds)), 'r--')
            plt.savefig(filename)

            # plot the tolerance parameter of the adaptive poisoning attack
            plt.clf()
            if args.poison_whole:
                filename = 'files/results/{}/whole-{}_ol_tol_params_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/results/{}/subpop-{}_ol_tol_params_for_real_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
            plt.plot(np.arange(len(ol_tol_params)), np.squeeze(np.array(ol_tol_params)), 'r--')
            plt.savefig(filename) 
        else:
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            data_info = np.load(filename)
            online_poisons_x = data_info["online_poisons_x"]
            online_poisons_y = data_info["online_poisons_y"]
            best_lower_bound = data_info["best_lower_bound"]
            conser_lower_bound = data_info["conser_lower_bound"]
            best_max_loss_x = data_info["best_max_loss_x"]
            best_max_loss_y = data_info["best_max_loss_y"]
            target_poison_max_losses = data_info["target_poison_max_losses"]
            total_loss_diffs = data_info["total_loss_diffs"]
            max_loss_diffs_reg = data_info["max_loss_diffs"]
            lower_bounds = data_info["lower_bounds"]
            ol_tol_params = data_info["ol_tol_params"]  
            ol_tol_par = ol_tol_params[-1]

            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            model_p_online = joblib.load(open(filename, 'rb'))

        ###  perform the KKT attack with same number of poisned points of our Adaptive attack ###
        if args.poison_whole:
            filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        if not os.path.isfile(filename):
            target_theta = best_target_theta
            if fit_intercept:
                target_bias = np.copy(best_target_bias)
            else:
                target_bias = 0
            total_epsilon = float(len(online_poisons_y))/X_train.shape[0]
            print("Number of poisons for kkt attack {}, the poison ratio {}".format(len(online_poisons_y),total_epsilon))
            model_dumb1.coef_ = np.array([target_theta])
            model_dumb1.intercept_ = np.array([target_bias]) 

            # sanity check for target model info
            print("--- Sanity Check Info for Subpop ---")
            margins = tst_sub_y*(tst_sub_x.dot(target_theta) + target_bias)
            _, ideal_target_err = calculate_loss(margins)
            margins =tst_nsub_y*(tst_nsub_x.dot(target_theta) + target_bias)
            _, ideal_collat_err = calculate_loss(margins)
            margins =y_test*(X_test.dot(target_theta) + target_bias)
            _, ideal_total_err = calculate_loss(margins)
            print("Ideal Total Test Acc:",1-ideal_total_err)
            print("Ideal Target Test Acc:",1-ideal_target_err)
            print("Ideal Collat Test Acc:",1-ideal_collat_err)
            margins = trn_sub_y*(trn_sub_x.dot(target_theta) + target_bias)
            _, ideal_target_err = calculate_loss(margins)
            margins =trn_nsub_y*(trn_nsub_x.dot(target_theta) + target_bias)
            _, ideal_collat_err = calculate_loss(margins)
            margins =y_train*(X_train.dot(target_theta) + target_bias)
            _, ideal_total_err = calculate_loss(margins)
            print("Ideal Total Train Acc:",1-ideal_total_err)
            print("Ideal Target Train Acc:",1-ideal_target_err)
            print("Ideal Collat Train Acc:",1-ideal_collat_err)

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
                x_neg_tuple=x_neg_tuple)

            target_grad = clean_grad_at_target_theta + ((1 + total_epsilon) * args.weight_decay * target_theta)
            epsilon_neg = (total_epsilon - target_bias_grad) / 2
            epsilon_pos = total_epsilon - epsilon_neg

            if (epsilon_neg >= 0) and (epsilon_neg <= total_epsilon):
                epsilon_pairs.append((epsilon_pos, epsilon_neg))

            for epsilon_pos in np.arange(0, total_epsilon + 1e-6, epsilon_increment):
                epsilon_neg = total_epsilon - epsilon_pos
                epsilon_pairs.append((epsilon_pos, epsilon_neg))
            
            for epsilon_pos, epsilon_neg in epsilon_pairs:
                print('\n## Trying epsilon_pos %s, epsilon_neg %s' % (epsilon_pos, epsilon_neg))
                X_modified, Y_modified, obj, x_pos, x, num_pos, num_neg = kkt_attack.kkt_attack(
                    two_class_kkt,
                    target_grad, target_theta,
                    total_epsilon * sub_frac, epsilon_pos * sub_frac, epsilon_neg * sub_frac,
                    X_train_cp, y_train_cp,
                    class_map, centroids, centroid_vec, sphere_radii, slab_radii,
                    target_bias, target_bias_grad, max_losses)
                
                # separate out the poisoned points
                idx_poison = slice(X_train.shape[0], X_modified.shape[0])
                idx_clean = slice(0, X_train.shape[0])
                
                X_poison = X_modified[idx_poison,:]
                Y_poison = Y_modified[idx_poison]   
                # unique points and labels in kkt attack
                unique_x, unique_indices, unique_counts = np.unique(X_poison,return_index = True,return_counts = True,axis=0)
                unique_y = Y_poison[unique_indices]               
                # retrain the model 
                C = 1.0 / (X_modified.shape[0] * args.weight_decay)
                model_p = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
                model_p.fit(X_modified, Y_modified)                 
                # acc on subpop and rest of pops
                trn_target_acc = model_p.score(trn_sub_x, trn_sub_y)
                trn_collat_acc = model_p.score(trn_nsub_x, trn_nsub_y)
                print()
                
                print('Test Total Acc : ', model_p.score(X_test, y_test))
                print('Test Target Acc : ', model_p.score(tst_sub_x, tst_sub_y))
                print('Test Collat Acc : ', model_p.score(tst_nsub_x, tst_nsub_y))
                print('Train Total Acc : ', model_p.score(X_train, y_train))
                print('Train Target Acc : ', trn_target_acc)
                print('Train Collat Acc : ', trn_collat_acc)

                # sanity check on the max loss difference between target model and kkt model
                kkt_tol_par = -1
                for y_b in set(y_train):
                    if y_b == 1:
                        max_loss_diff,_ = search_max_loss_pt(model_p,model_dumb1,y_b,x_pos_tuple,args)
                        if kkt_tol_par < max_loss_diff:
                            kkt_tol_par = max_loss_diff
                    elif y_b == -1:
                        max_loss_diff,_ = search_max_loss_pt(model_p,model_dumb1,y_b,x_neg_tuple,args)
                        if kkt_tol_par < max_loss_diff:
                            kkt_tol_par = max_loss_diff
                print("max loss difference between target and kkt model is:",kkt_tol_par)
                model_dumb1_b = model_dumb1.intercept_
                model_dumb1_b = model_dumb1_b[0]
                model_p_b = model_p.intercept_
                model_p_b = model_p_b[0]
                kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-model_p.coef_.reshape(-1))**2+(model_dumb1_b - model_p_b)**2)
                print("norm difference between target and kkt model is:",kkt_tol_par_norm)
                if trn_target_acc < best_target_acc:
                    best_target_acc = trn_target_acc
                    # used for theoretical lower bound computation
                    kkt_model_p.coef_ = np.copy(model_p.coef_)
                    kkt_model_p.intercept_ = np.copy(model_p.intercept_)
                    # best_target_acc1 = tst_target_acc
                    kkt_unique_x = np.copy(unique_x)
                    kkt_unique_y = np.copy(unique_y)
                    kkt_unique_counts = np.copy(unique_counts)
                    kkt_x_modified = np.copy(X_modified)
                    kkt_y_modified = np.copy(Y_modified)

            # save the model weights and the train and test data
            if args.poison_whole:
                filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            joblib.dump(kkt_model_p, filename)
            if args.poison_whole:
                filename = 'files/kkt_models/{}/whole-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/kkt_models/{}/subpop-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            best_kkt_theta = kkt_model_p.coef_.reshape(-1)
            best_kkt_bias = kkt_model_p.intercept_
            best_kkt_bias = best_kkt_bias[0]
            np.savez(filename,
                    kkt_x_modified = kkt_x_modified,
                    kkt_y_modified = kkt_y_modified,
                    kkt_unique_x = kkt_unique_x,
                    kkt_unique_y = kkt_unique_y,
                    kkt_unique_counts = kkt_unique_counts,
                    best_target_theta = best_target_theta,
                    best_target_bias = best_target_bias,
                    best_kkt_theta = best_kkt_theta,
                    best_kkt_bias = best_kkt_bias
                    )
        else:
            # load the kkt related model and data 
            if args.poison_whole:
                filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            kkt_model_p = joblib.load(open(filename, 'rb'))
            if args.poison_whole:
                filename = 'files/kkt_models/{}/whole-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/kkt_models/{}/subpop-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            data_info = np.load(filename)
            kkt_x_modified = data_info["kkt_x_modified"]
            kkt_y_modified = data_info["kkt_y_modified"]
            kkt_unique_x = data_info["kkt_unique_x"]
            kkt_unique_y = data_info["kkt_unique_y"]
            kkt_unique_counts = data_info["kkt_unique_counts"]
            # best_target_theta = data_info["best_target_theta"]
            # best_target_bias = data_info["best_target_bias"]
            idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
            idx_clean = slice(0, X_train.shape[0])
            total_epsilon = float(len(online_poisons_y))/X_train.shape[0]

            tst_target_acc = kkt_model_p.score(tst_sub_x, tst_sub_y)
            tst_collat_acc = kkt_model_p.score(tst_nsub_x, tst_nsub_y)
            print("--------Performance of Selected KKT attack model-------")
            # print('Total Train Acc : %.3f' % kkt_model_p.score(X_train, y_train))
            print('Total Test Acc : ', kkt_model_p.score(X_test, y_test))
            print('Test Target Acc : ', tst_target_acc)
            print('Test Collat Acc : ', tst_collat_acc)
            print('Total Train Acc : ', kkt_model_p.score(X_train, y_train))
            print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
            print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

        # sanity check for kkt points 
        assert np.array_equal(X_train, kkt_x_modified[idx_clean,:])
        assert np.array_equal(y_train,kkt_y_modified[idx_clean])

        print("min and max feature values of kkt points:")
        poison_xs = kkt_x_modified[idx_poison]
        poison_ys = kkt_y_modified[idx_poison]
        print(np.amax(poison_xs),np.amin(poison_xs))
        print(np.amax(poison_ys),np.amin(poison_ys))
        x_pos_min, x_pos_max = x_pos_tuple
        assert np.amax(poison_xs) <= (x_pos_max + float(x_pos_max)/100)
        assert np.amin(poison_xs) >= (x_pos_min - np.abs(float(x_pos_min))/100)

        # sanity check on the max loss difference between target model and kkt model
        print("----- Info of the Selected kkt model ---")
        print('Test Total Acc : ', kkt_model_p.score(X_test, y_test))
        print('Test Target Acc : ', kkt_model_p.score(tst_sub_x,tst_sub_y))
        print('Test Collat Acc : ', kkt_model_p.score(tst_nsub_x,tst_nsub_y))
        print('Train Total Acc : ', kkt_model_p.score(X_train, y_train))
        print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
        print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

        model_dumb1.coef_ = np.array([best_target_theta])
        model_dumb1.intercept_ = np.array([best_target_bias])
        kkt_tol_par = -1
        for y_b in set(y_train):
            if y_b == 1:
                max_loss_diff,_ = search_max_loss_pt(kkt_model_p,model_dumb1,y_b,x_pos_tuple,args)
                if kkt_tol_par < max_loss_diff:
                    kkt_tol_par = max_loss_diff
            elif y_b == -1:
                max_loss_diff,_ = search_max_loss_pt(kkt_model_p,model_dumb1,y_b,x_neg_tuple,args)
                if kkt_tol_par < max_loss_diff:
                    kkt_tol_par = max_loss_diff
        print("max loss difference between selected target and selected kkt model is:",kkt_tol_par)
        # assert np.array_equal(model_dumb1.coef_,target_model.coef_)
        # assert np.array_equal(model_dumb1.intercept_,target_model.intercept_)
        model_dumb1_b = model_dumb1.intercept_
        model_dumb1_b = model_dumb1_b[0]
        kkt_model_p_b = kkt_model_p.intercept_
        kkt_model_p_b = kkt_model_p_b[0]
        kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(model_dumb1_b - kkt_model_p_b)**2)
        print("norm difference between selected target and selected kkt model is:",kkt_tol_par_norm)
        if kkt_tol_par < 1e-4:
            print("something wrong with selected kkt model or the target model!")
            sys.exit(0)

        # compute the grad norm difference and store the value
        kkt_grad_norm_diff = compute_grad_norm_diff(best_target_theta,best_target_bias,total_epsilon,\
            X_train,y_train,poison_xs,poison_ys,args)
        ol_grad_norm_diff = compute_grad_norm_diff(best_target_theta,best_target_bias,total_epsilon,\
            X_train,y_train,online_poisons_x,online_poisons_y,args)
        print("Grad norm difference of KKT to target:",kkt_grad_norm_diff)
        print("Grad norm difference of adaptive to target:",ol_grad_norm_diff)
        
        # Print the lower bound and performance of different attacks 
        norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                            X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            subpop_data,
                                                            best_lower_bound,
                                                            conser_lower_bound,
                                                            kkt_tol_par,
                                                            ol_tol_par,
                                                            target_model,
                                                            kkt_model_p,
                                                            model_p_online,
                                                            len(online_poisons_y),
                                                            args)
        # key attack statistics are stored here
        real_target_lower_bound_and_attacks = real_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,len(online_poisons_y),len(online_poisons_y),
        kkt_tol_par, ol_tol_par] + norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores + [kkt_grad_norm_diff,ol_grad_norm_diff]
        # write key attack info to the csv files
        real_lower_bound_writer.writerow(real_target_lower_bound_and_attacks)

    if  args.target_model == "kkt" or args.target_model == "all":
        print("------- Use KKT model as Target Model -----")
        # load the kkt related model and data 
        if args.poison_whole:
            filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        kkt_model_p = joblib.load(open(filename, 'rb'))
        if args.poison_whole:
            filename = 'files/kkt_models/{}/whole-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/kkt_models/{}/subpop-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        data_info = np.load(filename)
        kkt_x_modified = data_info["kkt_x_modified"]
        kkt_y_modified = data_info["kkt_y_modified"]
        kkt_unique_x = data_info["kkt_unique_x"]
        kkt_unique_y = data_info["kkt_unique_y"]
        kkt_unique_counts = data_info["kkt_unique_counts"]
        # best_target_theta = data_info["best_target_theta"]
        # best_target_bias = data_info["best_target_bias"]
        idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
        idx_clean = slice(0, X_train.shape[0])

        tst_target_acc = kkt_model_p.score(tst_sub_x, tst_sub_y)
        tst_collat_acc = kkt_model_p.score(tst_nsub_x, tst_nsub_y)
        print("--------Performance of Selected KKT attack model-------")
        # print('Total Train Acc : %.3f' % kkt_model_p.score(X_train, y_train))
        print('Total Test Acc : ', kkt_model_p.score(X_test, y_test))
        print('Test Target Acc : ', tst_target_acc)
        print('Test Collat Acc : ', tst_collat_acc)
        print('Total Train Acc : ', kkt_model_p.score(X_train, y_train))
        print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
        print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

        target_model.coef_ = kkt_model_p.coef_
        target_model.intercept_ = kkt_model_p.intercept_
        kkt_tol_par = 0
        # check if adaptive online attack is needed
        if args.poison_whole:
            filename = 'files/online_models/{}/whole-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/online_models/{}/subpop-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        if not os.path.isfile(filename):
            # start the lower bound computation process
            online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
            best_max_loss_y, ol_tol_par, target_poison_max_losses, total_loss_diffs,\
            ol_tol_params, max_loss_diffs_reg, lower_bounds = incre_online_learning(X_train,
                                                                                    y_train,
                                                                                    curr_model,
                                                                                    target_model,
                                                                                    x_lim_tuples,
                                                                                    args,
                                                                                    ScikitModel,
                                                                                    target_model_type = "kkt",
                                                                                    attack_num_poison = kkt_x_modified.shape[0]-X_train.shape[0],
                                                                                    kkt_tol_par = kkt_tol_par)
            # retrain the online model based on poisons from our adaptive attack
            if len(online_poisons_y) > 0:
                online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
            else:
                print("online learning does not make progress and using original model!")
                online_poisons_x = np.array(online_poisons_x)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = X_train
                online_full_y = y_train
            print("shape of online poisoned points:",online_poisons_x.shape,online_poisons_y.shape)
            print("shape of full poisoned points:",online_full_x.shape,online_full_y.shape)
            # retrain the model based poisons from online learning
            C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
            fit_intercept = True
            model_p_online = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
            model_p_online.fit(online_full_x, online_full_y) 

            # need to save the posioned model from our attack, for the purpose of validating the lower bound for the online attack
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_kkt_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_kkt_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            joblib.dump(model_p_online, filename)
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            np.savez(filename,
                    online_poisons_x = online_poisons_x,
                    online_poisons_y = online_poisons_y,
                    best_lower_bound = best_lower_bound,
                    conser_lower_bound = conser_lower_bound,
                    best_max_loss_x = best_max_loss_x,
                    best_max_loss_y = best_max_loss_y,
                    target_poison_max_losses = target_poison_max_losses,
                    total_loss_diffs = total_loss_diffs, 
                    max_loss_diffs = max_loss_diffs_reg,
                    lower_bounds = lower_bounds,
                    ol_tol_params = ol_tol_params
                    )
        else:
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            data_info = np.load(filename)
            online_poisons_x = data_info["online_poisons_x"]
            online_poisons_y = data_info["online_poisons_y"]
            best_lower_bound = data_info["best_lower_bound"]
            conser_lower_bound = data_info["conser_lower_bound"]
            best_max_loss_x = data_info["best_max_loss_x"]
            best_max_loss_y = data_info["best_max_loss_y"]
            target_poison_max_losses = data_info["target_poison_max_losses"]
            total_loss_diffs = data_info["total_loss_diffs"]
            max_loss_diffs_reg = data_info["max_loss_diffs"]
            lower_bounds = data_info["lower_bounds"]
            ol_tol_params = data_info["ol_tol_params"]  
            ol_tol_par = ol_tol_params[-1]

            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_kkt_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_kkt_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            model_p_online = joblib.load(open(filename, 'rb'))

        # summarize the attack results
        norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                            X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            subpop_data,
                                                            best_lower_bound,
                                                            conser_lower_bound,
                                                            kkt_tol_par,
                                                            ol_tol_par,
                                                            target_model,
                                                            kkt_model_p,
                                                            model_p_online,
                                                            kkt_x_modified.shape[0]-X_train.shape[0],
                                                            args)
        if best_lower_bound > (kkt_x_modified.shape[0]-X_train.shape[0]):
            print("violation observed for the lower bound of KKT model!")
            sys.exit(0)
        kkt_target_lower_bound_and_attacks = kkt_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,\
            kkt_x_modified.shape[0]-X_train.shape[0],len(online_poisons_y),
        kkt_tol_par, ol_tol_par] + norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores
        # write key attack info to the csv files
        kkt_lower_bound_writer.writerow(kkt_target_lower_bound_and_attacks)
        ## draw the trend and also save some important statistics ##
        # just plot the total loss difference variation and max loss diff variation
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_max_loss_diff_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_max_loss_diff_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(max_loss_diffs_reg)), np.squeeze(np.array(max_loss_diffs_reg)), 'r--')
        plt.savefig(filename)   

        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_total_loss_diff_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_total_loss_diff_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)

        plt.plot(np.arange(len(total_loss_diffs)), np.squeeze(np.array(total_loss_diffs)), 'r--')
        plt.savefig(filename)  
        # plot the curve of lower bound and loss value w.r.t. iterations, also save these values for future use
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_lower_bound_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_lower_bound_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(lower_bounds)), np.squeeze(np.array(lower_bounds)), 'r--')
        plt.savefig(filename)

        # plot the tolerance parameter of the adaptive poisoning attack
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_ol_tol_params_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_ol_tol_params_for_kkt_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(ol_tol_params)), np.squeeze(np.array(ol_tol_params)), 'r--')
        plt.savefig(filename) 
    if args.target_model == "ol" or args.target_model == "all":
        print("------- Use Aptive Poison model as Target Model -----")
        # load the adaptive attack models
        if args.poison_whole:
            filename = 'files/online_models/{}/whole-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/online_models/{}/subpop-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        target_model_p_online = joblib.load(open(filename, 'rb'))
        if args.poison_whole:
            filename = 'files/online_models/{}/whole-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/online_models/{}/subpop-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        data_info = np.load(filename)
        target_online_poisons_x = data_info["online_poisons_x"]
        target_online_poisons_y = data_info["online_poisons_y"]
        target_online_full_x = np.concatenate((X_train,target_online_poisons_x),axis = 0)
        target_online_full_y = np.concatenate((y_train,target_online_poisons_y),axis = 0)

        if args.poison_whole:
            filename = 'files/online_models/{}/whole-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/online_models/{}/subpop-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        if not os.path.isfile(filename):
            # validate it only when adaptive attack did execute for a while
            if len(target_online_poisons_y) > 0:
                # separate out the poisoned points
                idx_poison = slice(X_train.shape[0], target_online_full_x.shape[0])
                idx_clean = slice(0, X_train.shape[0])
                assert np.array_equal(X_train, target_online_full_x[idx_clean,:])
                assert np.array_equal(y_train, target_online_full_y[idx_clean])    
                target_model.coef_ = target_model_p_online.coef_
                target_model.intercept_ = target_model_p_online.intercept_
                # load the kkt model and get the kkt stop criteria
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                kkt_model_p = joblib.load(open(filename, 'rb'))
                kkt_tol_par_max_loss = -1
                for y_b in set(y_train):
                    if y_b == 1:
                        max_loss_diff,_ = search_max_loss_pt(kkt_model_p,target_model,y_b,x_pos_tuple,args)
                        if kkt_tol_par_max_loss < max_loss_diff:
                            kkt_tol_par_max_loss = max_loss_diff
                    elif y_b == -1:
                        max_loss_diff,_ = search_max_loss_pt(kkt_model_p,target_model,y_b,x_neg_tuple,args)
                        if kkt_tol_par_max_loss < max_loss_diff:
                            kkt_tol_par_max_loss = max_loss_diff
                model_dumb1_b = model_dumb1.intercept_
                model_dumb1_b = model_dumb1_b[0]
                kkt_model_p_b = kkt_model_p.intercept_
                kkt_model_p_b = kkt_model_p_b[0]
                kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(model_dumb1_b - kkt_model_p_b)**2)
                if args.online_alg_criteria == "max_loss":
                    kkt_tol_par = kkt_tol_par_max_loss
                elif args.online_alg_criteria == "norm":
                    kkt_tol_par = kkt_tol_par_norm
                print("max loss and norm criterias of kkt attack:",kkt_tol_par_max_loss,kkt_tol_par_norm)

                # start the evaluation process
                online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
                best_max_loss_y, ol_tol_par, target_poison_max_losses, total_loss_diffs,\
                ol_tol_params, max_loss_diffs_reg, lower_bounds = incre_online_learning(X_train,
                                                                                        y_train,
                                                                                        curr_model,
                                                                                        target_model,
                                                                                        x_lim_tuples,
                                                                                        args,
                                                                                        ScikitModel,
                                                                                        target_model_type = "ol",
                                                                                        attack_num_poison = len(target_online_poisons_y),
                                                                                        kkt_tol_par = kkt_tol_par)
                # retrain the online model based on poisons from our adaptive attack
                if len(online_poisons_y) > 0:
                    online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                    online_poisons_y = np.array(online_poisons_y)
                    online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                    online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
                else:
                    print("online learning does not make progress and using original model!")
                    online_poisons_x = np.array(online_poisons_x)
                    online_poisons_y = np.array(online_poisons_y)
                    online_full_x = X_train
                    online_full_y = y_train
                print("shape of online poisoned points:",online_poisons_x.shape,online_poisons_y.shape)
                print("shape of full poisoned points:",online_full_x.shape,online_full_y.shape)
                # retrain the model based poisons from online learning
                C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
                fit_intercept = True
                model_p_online = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
                model_p_online.fit(online_full_x, online_full_y) 
                # need to save the posioned model from our attack, for the purpose of validating the lower bound for the online attack
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_online_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_online_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                joblib.dump(model_p_online, filename)
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                np.savez(filename,
                        online_poisons_x = online_poisons_x,
                        online_poisons_y = online_poisons_y,
                        best_lower_bound = best_lower_bound,
                        conser_lower_bound = conser_lower_bound,
                        best_max_loss_x = best_max_loss_x,
                        best_max_loss_y = best_max_loss_y,
                        target_poison_max_losses = target_poison_max_losses,
                        total_loss_diffs = total_loss_diffs, 
                        max_loss_diffs = max_loss_diffs_reg,
                        lower_bounds = lower_bounds,
                        ol_tol_params = ol_tol_params
                        ) 
        else:
            # load data
            if args.poison_whole:
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            data_info = np.load(filename)
            online_poisons_x = data_info["online_poisons_x"]
            online_poisons_y = data_info["online_poisons_y"]
            best_lower_bound = data_info["best_lower_bound"]
            conser_lower_bound = data_info["conser_lower_bound"]
            best_max_loss_x = data_info["best_max_loss_x"]
            best_max_loss_y = data_info["best_max_loss_y"]
            target_poison_max_losses = data_info["target_poison_max_losses"]
            total_loss_diffs = data_info["total_loss_diffs"]
            max_loss_diffs_reg = data_info["max_loss_diffs"]
            lower_bounds = data_info["lower_bounds"]
            ol_tol_params = data_info["ol_tol_params"] 
            ol_tol_par = ol_tol_params[-1] 
            # load model
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_online_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_online_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            model_p_online = joblib.load(open(filename, 'rb'))

        # summarize attack results
        norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                            X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            subpop_data,
                                                            best_lower_bound,
                                                            conser_lower_bound,
                                                            kkt_tol_par,
                                                            ol_tol_par,
                                                            target_model,
                                                            kkt_model_p,
                                                            model_p_online,
                                                            len(target_online_poisons_y),
                                                            args)
        if best_lower_bound > len(target_online_poisons_y):
            print("violation observed for the lower bound of Adaptive attack model!")
            sys.exit(0)
        ol_target_lower_bound_and_attacks = ol_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,len(target_online_poisons_y),len(online_poisons_y),
                kkt_tol_par, ol_tol_par] + norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores
        # write to csv files
        ol_lower_bound_writer.writerow(ol_target_lower_bound_and_attacks)

        ## draw the trend and also save some important statistics ##
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_max_loss_diff_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_max_loss_diff_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(max_loss_diffs_reg)), np.squeeze(np.array(max_loss_diffs_reg)), 'r--')
        plt.savefig(filename)   

        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_total_loss_diff_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_total_loss_diff_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)

        plt.plot(np.arange(len(total_loss_diffs)), np.squeeze(np.array(total_loss_diffs)), 'r--')
        plt.savefig(filename)  
        # plot the curve of lower bound and loss value w.r.t. iterations, also save these values for future use
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_lower_bound_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_lower_bound_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(lower_bounds)), np.squeeze(np.array(lower_bounds)), 'r--')
        plt.savefig(filename)

        # plot the tolerance parameter of the adaptive poisoning attack
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_ol_tol_params_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_ol_tol_params_for_online_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(ol_tol_params)), np.squeeze(np.array(ol_tol_params)), 'r--')
        plt.savefig(filename)  

    if args.target_model == "compare":
        # it makes more sense to compare KKT and our adaptive attack with same number poisoned points
            # load the kkt model and get the kkt stop criteria
        print("------- Use Actual target model as Target Model, for comparison purpose -----")          
        if args.poison_whole:
            filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
        kkt_model_p = joblib.load(open(filename, 'rb'))
        if args.poison_whole:
            filename = 'files/kkt_models/{}/whole-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/kkt_models/{}/subpop-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        data_info = np.load(filename)
        kkt_x_modified = data_info["kkt_x_modified"]
        idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
        # reload the target classifier as actual target classifier
        target_model.coef_= np.array([best_target_theta])
        target_model.intercept_ = np.array([best_target_bias])
        # get the kkt stop criteria 
        kkt_tol_par_max_loss = -1
        for y_b in set(y_train):
            if y_b == 1:
                max_loss_diff,_ = search_max_loss_pt(kkt_model_p,target_model,y_b,x_pos_tuple,args)
                if kkt_tol_par_max_loss < max_loss_diff:
                    kkt_tol_par_max_loss = max_loss_diff
            elif y_b == -1:
                max_loss_diff,_ = search_max_loss_pt(kkt_model_p,target_model,y_b,x_neg_tuple,args)
                if kkt_tol_par_max_loss < max_loss_diff:
                    kkt_tol_par_max_loss = max_loss_diff

        target_model_b = target_model.intercept_
        target_model_b = target_model_b[0]
        kkt_model_p_b = kkt_model_p.intercept_
        kkt_model_p_b = kkt_model_p_b[0]
        kkt_tol_par_norm = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(target_model_b - kkt_model_p_b)**2)
        # kkt_tol_par_norm = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-kkt_model_p.coef_)**2+(target_model.intercept_.reshape(-1) - kkt_model_p.intercept_)**2)
        if args.online_alg_criteria == "max_loss":
            kkt_tol_par = kkt_tol_par_max_loss
        elif args.online_alg_criteria == "norm":
            kkt_tol_par = kkt_tol_par_norm
        print("max loss and norm criterias of kkt attack:",kkt_tol_par_max_loss,kkt_tol_par_norm)
        if kkt_tol_par < 1e-4:
            print("something wrong with selected kkt model or the target model!")
            sys.exit(0)

        # check if it is necessary to conduct the adaptive poisoning attack
        if args.poison_whole:
            filename = 'files/online_models/{}/whole-{}_online_for_compare_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/online_models/{}/subpop-{}_online_for_compare_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
        if not os.path.isfile(filename):
            # target_model.coef_= np.array([best_target_theta])
            # target_model.intercept_ = np.array([best_target_bias])
            # start the evaluation process
            online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
            best_max_loss_y, ol_tol_par, target_poison_max_losses, total_loss_diffs,\
            ol_tol_params, max_loss_diffs_reg, lower_bounds = incre_online_learning(X_train,
                                                                                    y_train,
                                                                                    curr_model,
                                                                                    target_model,
                                                                                    x_lim_tuples,
                                                                                    args,
                                                                                    ScikitModel,
                                                                                    target_model_type = "compare",
                                                                                    attack_num_poison = 0,
                                                                                    kkt_tol_par = kkt_tol_par)
            # retrain the online model based on poisons from our adaptive attack
            if len(online_poisons_y) > 0:
                online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
            else:
                print("online learning does not make progress and using original model!")
                online_poisons_x = np.array(online_poisons_x)
                online_poisons_y = np.array(online_poisons_y)
                online_full_x = X_train
                online_full_y = y_train
            print("shape of online poisoned points:",online_poisons_x.shape,online_poisons_y.shape)
            print("shape of full poisoned points:",online_full_x.shape,online_full_y.shape)
            # retrain the model based poisons from online learning
            C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
            fit_intercept = True
            model_p_online = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
            model_p_online.fit(online_full_x, online_full_y) 

            # need to save the posioned model from our attack, for the purpose of validating the lower bound for the online attack
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_compare_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_compare_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            joblib.dump(model_p_online, filename)
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_compare_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_compare_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            np.savez(filename,
                    online_poisons_x = online_poisons_x,
                    online_poisons_y = online_poisons_y,
                    best_lower_bound = best_lower_bound,
                    conser_lower_bound = conser_lower_bound,
                    best_max_loss_x = best_max_loss_x,
                    best_max_loss_y = best_max_loss_y,
                    target_poison_max_losses = target_poison_max_losses,
                    total_loss_diffs = total_loss_diffs, 
                    max_loss_diffs = max_loss_diffs_reg,
                    lower_bounds = lower_bounds,
                    ol_tol_params = ol_tol_params
                    )
        else:
            # load data
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_compare_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_compare_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            data_info = np.load(filename)
            online_poisons_x = data_info["online_poisons_x"]
            online_poisons_y = data_info["online_poisons_y"]
            best_lower_bound = data_info["best_lower_bound"]
            conser_lower_bound = data_info["conser_lower_bound"]
            best_max_loss_x = data_info["best_max_loss_x"]
            best_max_loss_y = data_info["best_max_loss_y"]
            target_poison_max_losses = data_info["target_poison_max_losses"]
            total_loss_diffs = data_info["total_loss_diffs"]
            max_loss_diffs_reg = data_info["max_loss_diffs"]
            lower_bounds = data_info["lower_bounds"]
            ol_tol_params = data_info["ol_tol_params"] 
            ol_tol_par = ol_tol_params[-1] 
            # load model
            if args.poison_whole:
                filename = 'files/online_models/{}/whole-{}_online_for_compare_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_for_compare_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            model_p_online = joblib.load(open(filename, 'rb'))
        # summarize attack results
        norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                            X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            subpop_data,
                                                            best_lower_bound,
                                                            conser_lower_bound,
                                                            kkt_tol_par,
                                                            ol_tol_par,
                                                            target_model,
                                                            kkt_model_p,
                                                            model_p_online,
                                                            kkt_x_modified.shape[0]-X_train.shape[0],
                                                            args)
                                                
        compare_target_lower_bound_and_attacks = compare_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,\
            kkt_x_modified.shape[0]-X_train.shape[0],len(online_poisons_y),
        kkt_tol_par, ol_tol_par] + norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores
        # write to csv file
        compare_lower_bound_writer.writerow(compare_target_lower_bound_and_attacks)
        ## draw the trend and also save some important statistics ##
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/compare_whole-{}_max_loss_diff_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/compare_subpop-{}_max_loss_diff_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(max_loss_diffs_reg)), np.squeeze(np.array(max_loss_diffs_reg)), 'r--')
        plt.savefig(filename)   

        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/compare_whole-{}_total_loss_diff_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/compare_subpop-{}_total_loss_diff_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)

        plt.plot(np.arange(len(total_loss_diffs)), np.squeeze(np.array(total_loss_diffs)), 'r--')
        plt.savefig(filename)  
        # plot the curve of lower bound and loss value w.r.t. iterations, also save these values for future use
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/compare_whole-{}_lower_bound_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/compare_subpop-{}_lower_bound_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(lower_bounds)), np.squeeze(np.array(lower_bounds)), 'r--')
        plt.savefig(filename)
    
        # plot the tolerance parameter of the adaptive poisoning attack
        plt.clf()
        if args.poison_whole:
            filename = 'files/results/{}/whole-{}_ol_tol_params_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        else:
            filename = 'files/results/{}/subpop-{}_ol_tol_params_for_compare_tol-{}.png'.format(dataset_name,cl_ind,args.incre_tol_par)
        plt.plot(np.arange(len(ol_tol_params)), np.squeeze(np.array(ol_tol_params)), 'r--')
        plt.savefig(filename)  

# close all files
if args.target_model == "all":
    kkt_lower_bound_file.flush()
    kkt_lower_bound_file.close()
    real_lower_bound_file.flush()
    real_lower_bound_file.close()
    ol_lower_bound_file.flush()
    ol_lower_bound_file.close()
    # compare_lower_bound_file.flush()
    # compare_lower_bound_file.close()
elif args.target_model == "kkt":
    kkt_lower_bound_file.flush()
    kkt_lower_bound_file.close()
elif args.target_model == "real":
    real_lower_bound_file.flush()
    real_lower_bound_file.close()
elif args.target_model == "ol":
    ol_lower_bound_file.flush()
    ol_lower_bound_file.close()
elif args.target_model == "compare":
    compare_lower_bound_file.flush()
    compare_lower_bound_file.close()
