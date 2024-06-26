from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn import linear_model, svm
# from utils import *
from utils import svm_model, logistic_model, calculate_loss, dist_to_boundary, get_subpop_inds
import pickle
import argparse
from datasets import load_dataset
import os
import scipy
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or rlogistic regression')
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, 2d_toy, dogfish")
parser.add_argument('--weight_decay',default=0.09, type=float, help='weight decay for regularizers')
parser.add_argument('--improved',action="store_true",help='if true, target classifier is obtained through improved process')
parser.add_argument('--subpop',action="store_true",help='if true, subpopulation attack will be performed')
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature', 'random'], help='subpopulaton type: cluster, feature, or random')
parser.add_argument('--all_subpops', action="store_true", help='if true, generates models for all subpopulations')
parser.add_argument('--min_d2db', default=None, type=float, help='acceptable minimum distance to decision boundary')
parser.add_argument('--interp', default=None, type=float, help='linear interpolation between clean and target model')
parser.add_argument('--valid_theta_err', default=None, type=float, help='minimum target model classification error')
parser.add_argument('--selection_criteria', default='min_collateral', choices=['min_collateral', 'max_loss'], help='target model choice criteria')
parser.add_argument('--save_all', action='store_true', help='if true, saves all valid generated target models')

args = parser.parse_args()

# if true, only select the best target classifier for each subpop
# for subpop attack: it is the one with 0% acc on subpop (train) and minimal damage on rest of pop (train)
# for indiscriminative attack, it is the one with highest train error
select_best = not args.save_all

# whether poisoning attack is targeted or indiscriminative
if args.dataset == "adult":
    subpop = True
elif args.dataset == "mnist_17":
    subpop = True
elif args.dataset == "2d_toy":
    # this is only to help test the optimzation framework
    # for generating the target classifier
    subpop = True
elif args.dataset == "dogfish":
    subpop = args.subpop
    args.weight_decay = 1.1
elif args.dataset in ['loan', 'compas', 'synthetic']:
    subpop = True
else:
    subpop = args.subpop

if args.model_type == 'svm':
    ScikitModel = svm_model
else:
    ScikitModel = logistic_model

# reduce number of searches on target classifier
prune_theta = True

dataset_name = args.dataset
assert dataset_name in ['adult','mnist_17','2d_toy','dogfish', 'loan', 'compas', 'synthetic']

subpop_type = args.subpop_type

# load data
X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)

if min(Y_test)>-1:
    Y_test = 2*Y_test-1
if min(Y_train) > -1:
    Y_train = 2*Y_train - 1

max_iter = -1

fit_intercept = True

C = 1.0 / (X_train.shape[0] * args.weight_decay)
model = ScikitModel(
    C=C,
    tol=1e-10,
    fit_intercept=fit_intercept,
    random_state=24,
    verbose=False,
    max_iter = 10000)
model.fit(X_train, Y_train)
orig_theta = model.coef_.reshape(-1)
orig_bias = model.intercept_

# calculate the clean model acc
train_acc = model.score(X_train,Y_train)
test_acc = model.score(X_test,Y_test)

margins = Y_train*(X_train.dot(orig_theta) + orig_bias)
train_loss, train_err = calculate_loss(margins)
clean_train_loss = train_loss
clean_total_train_loss = train_loss*X_train.shape[0]
# test margins and loss
margins = Y_test*(X_test.dot(orig_theta) + orig_bias)
test_loss, test_err = calculate_loss(margins)
clean_total_test_loss = test_loss*X_test.shape[0]

if not subpop:
    ym = (-1)*Y_test
    if args.dataset in ['mnist_17','dogfish']:
        # loss percentile and repeated points, used for indiscriminative attack
        if args.dataset == 'mnist_17':
            valid_theta_errs = [0.05,0.1,0.15]
        else:
            valid_theta_errs = [0.1,0.2,0.3,0.4]
        quantile_tape = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,0.6]
        rep_tape = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30,50,60]
        # valid_theta_errs = [0.5,0.6,0.7,0.8]
        # quantile_tape = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,0.6,0.7,0.8,0.9]
        # rep_tape = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30,40,60,70,80,90,100]

    elif args.dataset == '2d_toy':
        valid_theta_errs = [0.1,0.15]
        # loss percentile and repeated points, used for indiscriminative attack
        quantile_tape = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,0.6,0.65,0.75,0.8,0.85,0.9]
        rep_tape = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30,35,40]
    # valid_theta_errs = [0.2]
# we prefer points with lower loss (higher loss in correct labels)
clean_margins = Y_test*(X_test.dot(orig_theta) + orig_bias)

y_list = [1,-1]
if not subpop:
    # procedure of generating target classifier, refer to strong poisoning attack paper
    # however, we assume adversaries do not have access to test data
    thetas = []
    biases = []
    train_losses = []
    test_errs = []
    collat_errs = []
    num_poisons = []
    for loss_quantile in quantile_tape:
        for tar_rep in rep_tape:
            X_tar = []
            Y_tar = []
            margin_thresh = np.quantile(clean_margins, loss_quantile)
            for i in range(len(y_list)):
                active_cur = np.logical_and(Y_test == y_list[i],clean_margins < margin_thresh)
                X_tar_cur = X_test[active_cur,:]
                y_tar_cur = ym[active_cur]
                # y_orig_cur = Y_test[active_cur]
                X_tar.append(X_tar_cur)
                Y_tar.append(y_tar_cur)
                # Y_orig = Y_orig.append(y_orig_cur)
            X_tar = np.concatenate(X_tar, axis=0)
            Y_tar = np.concatenate(Y_tar, axis=0)
            # repeat points
            X_tar = np.repeat(X_tar, tar_rep, axis=0)
            Y_tar = np.repeat(Y_tar, tar_rep, axis=0)
            X_train_p = np.concatenate((X_train,X_tar),axis = 0)
            Y_train_p = np.concatenate((Y_train,Y_tar),axis = 0)
            # build another model for poisoned points
            C = 1.0 / (X_train_p.shape[0] * args.weight_decay)
            model_p = ScikitModel(
                    C=C,
                    tol=1e-10,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 10000)
            model_p.fit(X_train_p,Y_train_p)
            target_theta, target_bias = model_p.coef_.reshape(-1), model_p.intercept_
            # train margin and loss
            margins = Y_train_p*(X_train_p.dot(target_theta) + target_bias)
            train_loss, train_err = calculate_loss(margins)
            train_acc = model_p.score(X_train_p,Y_train_p)
            margins = Y_train*(X_train.dot(target_theta) + target_bias)
            train_loss, train_err = calculate_loss(margins)
            # use the regularized loss function
            train_loss = train_loss + (args.weight_decay/2) * np.linalg.norm(target_theta)**2

            train_acc = model_p.score(X_train,Y_train)
            # test margins and loss
            # # here, we replace test loss with train loss because we cannot use test loss
            # # to prune the theta, see below
            margins = Y_test*(X_test.dot(target_theta) + target_bias)
            test_loss, test_err = calculate_loss(margins)
            test_acc = model_p.score(X_test,Y_test)
            # improved attack actually generates the poisoned points based current model
            if args.improved:
                clean_margins = Y_test*(X_test.dot(target_theta) + target_bias)
            # collect the info
            thetas.append(target_theta)
            biases.append(target_bias[0])
            train_losses.append(train_loss)
            test_errs.append(test_err)
            collat_errs.append(test_err)
            num_poisons.append(len(Y_tar))
    thetas = np.array(thetas)
    biases = np.array(biases)
    train_losses = np.array(train_losses)
    test_errs = np.array(test_errs)
    collat_errs = np.array(collat_errs)
    num_poisons = np.array(num_poisons)

    # Prune away target parameters that are not on the Pareto boundary of (train_loss, test_error)
    if prune_theta:
        # use the one with lowest acc and lowest loss on train data
        # best theta is selected as one satisfy the attack goal and lowest loss on clean train data
        negtest_errs = [-x for x in test_errs]
        iisort = np.argsort(np.array(negtest_errs))
        best_theta_ids = []
        # select the best target classifier
        for valid_theta_err in valid_theta_errs:
            min_train_loss = 1e9
            for ii in iisort:
                if test_errs[ii] > valid_theta_err:
                    if train_losses[ii] < min_train_loss:
                        best_theta_idx = ii
                        min_train_loss = train_losses[ii]
            best_theta_ids.append(best_theta_idx)

        # pruned all the thetas
        iisort_pruned = []
        ids_remain = []
        min_train_loss = 1e9
        for ii in iisort:
            if train_losses[ii] < min_train_loss:
                iisort_pruned.append(ii)
                min_train_loss = train_losses[ii]
        pruned_thetas = thetas[iisort_pruned]
        pruned_biases = biases[iisort_pruned]
        pruned_train_losses = train_losses[iisort_pruned]
        pruned_test_errs = test_errs[iisort_pruned]
        prunned_collat_errs = collat_errs[iisort_pruned]

    # save all params together
    data_all = {}
    data_all['thetas'] = thetas
    data_all['biases'] = biases
    data_all['train_losses'] = train_losses
    data_all['test_errs'] = test_errs
    data_all['collat_errs'] = collat_errs

    data_pruned = {}
    data_pruned['thetas'] = pruned_thetas
    data_pruned['biases'] = pruned_biases
    data_pruned['train_losses'] = pruned_train_losses
    data_pruned['test_errs'] = pruned_test_errs
    data_pruned['collat_errs'] = prunned_collat_errs

    # best_theta = thetas[iisort_pruned[0]]
    # best_bias = biases[iisort_pruned[0]]
    # best_train_loss = train_losses[iisort_pruned[0]]
    # best_test_err = test_errs[iisort_pruned[0]]
    # best_collat_err = collat_errs[iisort_pruned[0]]

    if select_best:
        for kkk in range(len(valid_theta_errs)):
            valid_theta_err = valid_theta_errs[kkk]
            best_theta_idx = best_theta_ids[kkk]

            best_theta = thetas[best_theta_idx]
            best_bias = biases[best_theta_idx]
            best_train_loss = train_losses[best_theta_idx]
            best_test_err = test_errs[best_theta_idx]
            best_collat_err = collat_errs[best_theta_idx]
            best_num_poison = num_poisons[best_theta_idx]

            data_best = {}
            data_best['thetas'] = best_theta
            data_best['biases'] = best_bias
            data_best['train_losses'] = best_train_loss
            data_best['test_errs'] = best_test_err
            data_best['collat_errs'] = best_collat_err

            # choose the one with least train error
            if not os.path.isdir('files/target_classifiers/{}/{}'.format(dataset_name,args.model_type)):
                os.makedirs('files/target_classifiers/{}/{}'.format(dataset_name,args.model_type))
            if args.improved:
                file_all = open('files/target_classifiers/{}/{}/improved_best_theta_whole_err-{}'.format(dataset_name,args.model_type,valid_theta_err), 'wb')
            else:
                file_all = open('files/target_classifiers/{}/{}/orig_best_theta_whole_err-{}'.format(dataset_name,args.model_type,valid_theta_err), 'wb')
            # dump information to that file
            pickle.dump(data_best, file_all,protocol=2)
            file_all.close()
    else:
        if not os.path.isdir('files/target_classifiers/{}/{}'.format(dataset_name,args.model_type)):
            os.makedirs('files/target_classifiers/{}/{}'.format(dataset_name,args.model_type))
        if args.improved:
            file_all = open('files/target_classifiers/{}/{}/improved_thetas_whole'.format(dataset_name,args.model_type), 'wb')
            file_pruned = open('files/target_classifiers/{}/{}/improved_thetas_whole_pruned'.format(dataset_name,args.model_type), 'wb')
        else:
            file_all = open('files/target_classifiers/{}/{}/orig_thetas_whole'.format(dataset_name,args.model_type), 'wb')
            file_pruned = open('files/target_classifiers/{}/{}/orig_thetas_whole_pruned'.format(dataset_name,args.model_type), 'wb')

        # dump information to that file
        pickle.dump(data_all, file_all,protocol=2)
        file_all.close()
        # save pruned thetas
        # dump information to that file
        pickle.dump(data_pruned, file_pruned,protocol=2)
        file_pruned.close()
elif args.improved:
    # load and attack each subpopulation
    # generation process for subpop: directly flip the labels of subpop
    # choose 5 with highest original acc

    # find the clusters and corresponding subpop size
    trn_subpop_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
    with open(trn_subpop_fname, 'r') as f:
        trn_all_subpops = [np.array(map(int, line.split())) for line in f]
    tst_subpop_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
    with open(tst_subpop_fname, 'r') as f:
        tst_all_subpops = [np.array(map(int, line.split())) for line in f]

    pois_rates = [0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9,1.0,1.1,1.2,1.3,1.4,1.5]

    subpops_flattened = np.concatenate(trn_all_subpops).flatten()
    subpop_inds, subpop_cts = np.unique(subpops_flattened, return_counts=True)

    trn_sub_accs = []
    for i in range(len(subpop_cts)):
        subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
        tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
        trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
        # indices of points belong to cluster
        tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train)

        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
        tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
        trn_sub_acc = model.score(trn_sub_x, trn_sub_y)
        trn_sub_accs.append(trn_sub_acc)

    # sort the subpop based on tst acc and choose 5 highest ones
    if args.dataset in ['adult','dogfish']:
        highest_5_inds = np.argsort(trn_sub_accs)[-3:]
    elif args.dataset == '2d_toy':
        highest_5_inds = np.argsort(trn_sub_accs)[-4:]
    subpop_inds = subpop_inds[highest_5_inds]
    subpop_cts = subpop_cts[highest_5_inds]

    # save the selected subpop info
    cls_fname = 'files/data/{}_{}_selected_subpops.txt'.format(dataset_name, args.model_type)

    np.savetxt(cls_fname,np.array([subpop_inds,subpop_cts]), fmt=u'%i'.encode('utf-8'))

    if dataset_name == 'dogfish':
        valid_theta_errs = [0.9]
    else:
        valid_theta_errs = [1.0]

    choose_best = True

    for valid_theta_err in valid_theta_errs:
        for i in range(len(subpop_cts)):
            subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
            thetas = []
            biases = []
            train_losses = []
            test_errs = []
            collat_errs = []
            # best_collat_acc = 0
            if choose_best:
                min_train_loss = 1e10
            else:
                min_train_loss = 0

            tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
            trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
            # indices of points belong to cluster
            tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train)

            # get the corresponding points in the dataset
            tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
            tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
            trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
            trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
            tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
            # make sure subpop is from class -1
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

            # dist to decision boundary
            trn_sub_dist = dist_to_boundary(model.coef_.reshape(-1),model.intercept_,trn_sub_x)


            # try target generated with different ratios
            for kk in range(len(pois_rates)):
                pois_rate = pois_rates[kk]
                x_train_copy, y_train_copy = np.copy(X_train), np.copy(Y_train)
                pois_ct = int(pois_rate * X_train.shape[0])
                if pois_ct <= trn_sub_x.shape[0]:
                    pois_inds = np.argsort(trn_sub_dist)[:pois_ct]
                else:
                    pois_inds = np.random.choice(trn_sub_x.shape[0], pois_ct, replace=True)
                # generate the poisoning dataset by directly flipping labels
                pois_x, pois_y = trn_sub_x[pois_inds], -trn_sub_y[pois_inds]
                if pois_ct > trn_sub_x.shape[0]:
                    y_train_copy = np.delete(y_train_copy,trn_sbcl,axis=0)
                    x_train_copy = np.delete(x_train_copy,trn_sbcl,axis=0)
                    whole_y = np.concatenate((y_train_copy,pois_y),axis=0)
                    whole_x = np.concatenate((x_train_copy,pois_x),axis=0)
                else:
                    replace_idx = trn_sbcl[pois_inds]
                    y_train_copy[replace_idx] = -y_train_copy[replace_idx]
                    whole_x, whole_y = x_train_copy, y_train_copy

                # build another model for poisoned points
                C = 1.0 / (whole_x.shape[0]*args.weight_decay)
                model_p = ScikitModel(
                        C=C,
                        tol=1e-10,
                        fit_intercept=fit_intercept,
                        random_state=24,
                        verbose=False,
                        max_iter = 10000)

                model_p.fit(whole_x,whole_y)
                pois_acc = model_p.score(X_test,Y_test)
                trn_sub_acc = model_p.score(trn_sub_x, trn_sub_y)
                tst_sub_acc = model_p.score(tst_sub_x, tst_sub_y)
                trn_nsub_acc = model_p.score(trn_nsub_x,trn_nsub_y)

                # theta and bias of the model
                target_theta, target_bias = model_p.coef_.reshape(-1), model_p.intercept_

                margins = Y_train*(X_train.dot(target_theta) + target_bias)
                train_loss, _ = calculate_loss(margins)
                train_loss = train_loss + (args.weight_decay/2) * np.linalg.norm(target_theta)**2

                thetas.append(target_theta)
                biases.append(target_bias[0])
                # if trn_sub_acc == 0:
                #     if trn_nsub_acc > best_collat_acc:
                #         best_collat_acc = trn_nsub_acc
                #         best_theta = target_theta
                #         best_bias = target_bias[0]
                #         print("updated best collat train acc is:",trn_nsub_acc)

                # choose best valid classifier with lowest loss on clean data
                acc_threshold = 1 - valid_theta_err

                if tst_sub_acc <= acc_threshold:
                    if choose_best:
                        stop_cond = train_loss < min_train_loss
                    else:
                        stop_cond = train_loss > min_train_loss

                    if stop_cond:
                        min_train_loss = train_loss
                        best_theta = target_theta
                        best_bias = target_bias[0]
                        best_num_poisons = pois_ct
                        # if choose_best:
                        #     print("updated lowest train loss is:",train_loss)
                        # else:
                        #     print("updated highest train loss is:",train_loss)

            thetas = np.array(thetas)
            biases = np.array(biases)
            data_all = {}
            data_all['thetas'] = thetas
            data_all['biases'] = biases
            data_best = {}
            data_best['thetas'] = best_theta
            data_best['biases'] = best_bias

            # save all the target thetas
            if select_best:
                if not os.path.isdir('files/target_classifiers/{}/{}/{}'.format(dataset_name,args.model_type, subpop_type)):
                    os.makedirs('files/target_classifiers/{}/{}/{}'.format(dataset_name,args.model_type, subpop_type))
                file_all = open('files/target_classifiers/{}/{}/{}/improved_best_theta_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type,int(subpop_ind),valid_theta_err), 'wb')
                pickle.dump(data_best, file_all,protocol=2)
                file_all.close()
            else:
                if not os.path.isdir('files/target_classifiers/{}/{}/{}'.format(dataset_name,args.model_type, subpop_type)):
                    os.makedirs('files/target_classifiers/{}/{}/{}'.format(dataset_name,args.model_type, subpop_type))
                file_all = open('files/target_classifiers/{}/{}/{}/improved_thetas_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type, int(subpop_ind),valid_theta_err), 'wb')
                pickle.dump(data_all, file_all,protocol=2)
                file_all.close()
else:
    # load and attack each subpopulation
    # generation process for subpop: directly flip the labels of subpop
    # choose 5 with highest original acc

    # find the clusters and corresponding subpop size
    trn_subpop_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
    with open(trn_subpop_fname, 'r') as f:
        trn_all_subpops = [np.array(map(int, line.split())) for line in f]
    tst_subpop_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
    with open(tst_subpop_fname, 'r') as f:
        tst_all_subpops = [np.array(map(int, line.split())) for line in f]

    subpops_flattened = np.concatenate(trn_all_subpops).flatten()
    subpop_inds, subpop_cts = np.unique(subpops_flattened, return_counts=True)

    trn_sub_accs = []
    n_bad = 0
    for i in range(len(subpop_cts)):
        subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
        # indices of points belong to subpop
        tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
        trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
        tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train)

        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
        if len(tst_sub_y) == 0 or len(trn_sub_y) == 0:
            tst_sub_acc = -float('inf')
            trn_sub_acc = -float('inf')
            n_bad += 1
        else:
            tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
            trn_sub_acc = model.score(trn_sub_x, trn_sub_y)

        # check the target and collateral damage info
        trn_sub_accs.append(trn_sub_acc)

    # sort the subpop based on tst acc and choose 5 highest ones
    if args.all_subpops:
        highest_5_inds = np.argsort(trn_sub_accs)[n_bad:]
    elif args.dataset in ['adult','dogfish']:
        highest_5_inds = np.argsort(trn_sub_accs)[-3:]
    elif args.dataset == '2d_toy':
        highest_5_inds = np.argsort(trn_sub_accs)[-4:]
    else:
        highest_5_inds = np.argsort(trn_sub_accs)[-3:]
    subpop_inds = subpop_inds[highest_5_inds]
    subpop_cts = subpop_cts[highest_5_inds]

    # save the selected subpop info
    cls_fname = 'files/data/{}_{}_{}_selected_subpops.txt'.format(
        dataset_name, args.model_type, subpop_type)
    np.savetxt(cls_fname,np.array([subpop_inds,subpop_cts]), fmt=u'%i'.encode('utf-8'))

    if args.valid_theta_err is not None:
        valid_theta_errs = [args.valid_theta_err]
    elif dataset_name == 'dogfish':
        valid_theta_errs = [0.9]
    else:
        valid_theta_errs = [1.0]

    # repitition of the points
    if dataset_name == 'compas':
        quantile_tape = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55,0.8,1.0]
        rep_tape = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30,40,50,80,100,200,500]
    elif dataset_name != "dogfish":
        quantile_tape = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55,0.8,1.0]
        rep_tape = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30,40,50,80,100]
    else:
        quantile_tape = [0.40, 0.45, 0.50, 0.55,0.8,1.0]
        rep_tape = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30,40,50,80,100]

    # for subpop descriptions
    trn_desc_fname = 'files/data/{}_trn_{}_desc.csv'.format(dataset_name, subpop_type)
    trn_df = pd.read_csv(trn_desc_fname)

    for valid_theta_err in valid_theta_errs:
        for i in range(len(subpop_cts)):
            subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
            trn_df.loc[subpop_ind, 'Flip Num Poisons'] = float('inf')
            trn_df.loc[subpop_ind, 'Min Loss Diff'] = float('inf')
            thetas = []
            biases = []
            train_losses = []
            test_errs = []
            collat_errs = []
            # best_collat_acc = 0
            min_train_loss = 1e10
            max_test_target_loss = 0

            tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
            trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
            # indices of points belong to subpop
            tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train)

            # get the corresponding points in the dataset
            tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
            tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
            trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
            trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
            tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
            # make sure subpop is from class -1
            if dataset_name in ['adult', 'loan', 'compas', 'synthetic']:
                assert (tst_sub_y == -1).all()
                assert (trn_sub_y == -1).all()
            else:
                mode_tst = scipy.stats.mode(tst_sub_y)
                mode_trn = scipy.stats.mode(trn_sub_y)
                major_lab_trn = mode_trn.mode[0]
                major_lab_tst = mode_tst.mode[0]
                assert (tst_sub_y == major_lab_tst).all()
                assert (trn_sub_y == major_lab_tst).all()

            best_theta = None
            best_bias = None

            # find the loss percentile
            clean_margins = tst_sub_y*(tst_sub_x.dot(orig_theta) + orig_bias)
            ym = (-1)*tst_sub_y
            for loss_quantile in quantile_tape:
                rep_tape_tmp = rep_tape[:] # copies list, so we can modify it
                for tar_rep in rep_tape_tmp:
                    X_tar = []
                    Y_tar = []
                    margin_thresh = np.quantile(clean_margins, loss_quantile)
                    for i in range(len(y_list)):
                        active_cur = np.logical_and(tst_sub_y == y_list[i],clean_margins <= margin_thresh)
                        X_tar_cur = tst_sub_x[active_cur,:]
                        y_tar_cur = ym[active_cur]
                        # y_orig_cur = Y_test[active_cur]
                        X_tar.append(X_tar_cur)
                        Y_tar.append(y_tar_cur)
                        # Y_orig = Y_orig.append(y_orig_cur)
                    X_tar = np.concatenate(X_tar, axis=0)
                    Y_tar = np.concatenate(Y_tar, axis=0)

                    # repeat points
                    X_tar = np.repeat(X_tar, tar_rep, axis=0)
                    Y_tar = np.repeat(Y_tar, tar_rep, axis=0)
                    X_train_p = np.concatenate((X_train,X_tar),axis = 0)
                    Y_train_p = np.concatenate((Y_train,Y_tar),axis = 0)
                    # build another model for poisoned points
                    C = 1.0 / (X_train_p.shape[0] * args.weight_decay)
                    model_p = ScikitModel(
                            C=C,
                            tol=1e-10,
                            fit_intercept=fit_intercept,
                            random_state=24,
                            verbose=False,
                            max_iter = 10000)
                    model_p.fit(X_train_p,Y_train_p)
                    # plot the acc info
                    test_acc = model_p.score(X_test,Y_test)
                    train_acc = model_p.score(X_train,Y_train)
                    trn_sub_acc = model_p.score(trn_sub_x, trn_sub_y)
                    trn_nsub_acc = model_p.score(trn_nsub_x,trn_nsub_y)
                    tst_sub_acc = model_p.score(tst_sub_x, tst_sub_y)
                    tst_nsub_acc = model_p.score(tst_nsub_x, tst_nsub_y)

                    # theta and bias of the model
                    target_theta, target_bias = model_p.coef_.reshape(-1), model_p.intercept_

                    margins = Y_train*(X_train.dot(target_theta) + target_bias)
                    train_loss, _ = calculate_loss(margins)
                    train_loss = train_loss + (args.weight_decay/2) * np.linalg.norm(target_theta)**2
                    train_losses.append(train_loss)

                    margins = tst_sub_y*(tst_sub_x.dot(target_theta) + target_bias)
                    test_target_loss, test_error = calculate_loss(margins)
                    test_errs.append(test_error)

                    thetas.append(target_theta)
                    biases.append(target_bias[0])

                    margins = tst_nsub_y*(tst_nsub_x.dot(target_theta) + target_bias)
                    _, test_err = calculate_loss(margins)
                    collat_errs.append(test_err)

                    # choose best valid classifier with lowest loss on clean data
                    assert len(Y_tar) == len(X_train_p) - len(X_train)

                    acc_threshold = 1 - valid_theta_err

                    if tst_sub_acc <= acc_threshold:
                        trn_df.loc[subpop_ind, 'Min Loss Diff'] = min(trn_df.loc[subpop_ind, 'Min Loss Diff'], train_loss - clean_train_loss)
                        trn_df.loc[subpop_ind, 'Flip Num Poisons'] = min(trn_df.loc[subpop_ind, 'Flip Num Poisons'], len(Y_tar))
                        if args.selection_criteria == 'min_collateral':
                            if train_loss < min_train_loss:
                                min_train_loss = train_loss
                                best_theta = np.copy(target_theta)
                                best_bias = np.copy(target_bias[0])
                                best_num_poisons = np.copy(len(Y_tar))
                        elif args.selection_criteria == 'max_loss':
                            if test_target_loss > max_test_target_loss:
                                max_test_target_loss = test_target_loss
                                best_theta = np.copy(target_theta)
                                best_bias = np.copy(target_bias[0])
                                best_num_poisons = np.copy(len(Y_tar))

                    # if we never found a successful target model, try this:
                    if (best_theta is None) and (loss_quantile == quantile_tape[-1]) and (tar_rep == rep_tape_tmp[-1]):
                        assert tar_rep < 20000, "too many repetitions needed! giving up on subpop {}".format(subpop_ind)
                        rep_tape_tmp.append(2*rep_tape_tmp[-1])

            thetas = np.array(thetas)
            biases = np.array(biases)
            test_errs = np.array(test_errs)
            train_losses = np.array(train_losses)
            collat_errs = np.array(collat_errs)

            assert best_theta is not None, 'Was not able to find satisfactory target model!'

            # offset bias to satisfy min distance to boundary
            # assumes goal is to make positive
            norm_theta = np.linalg.norm(best_theta)
            near = np.min((np.dot(trn_sub_x,best_theta) + best_bias) / norm_theta)
            if args.min_d2db is not None and near < args.min_d2db:
                best_bias += norm_theta * (args.min_d2db - near)
                near = np.min(dist_to_boundary(best_theta, best_bias, trn_sub_x))
            if args.interp is not None and args.interp > 0:
                best_theta = args.interp*best_theta + (1. - args.interp)*orig_theta
                best_bias = args.interp*best_bias + (1. - args.interp)*orig_bias

            # discard non-optimal thetas
            if prune_theta:
                # prune by buckets:
                # for each accuracy threshold from acc_threshold...1.0,
                # take target theta with lowest loss on clean dataset

                ix_pruned = []

                acc_step = 0.05
                acc_buckets = np.arange(acc_threshold, 1.1, acc_step) # go a little over to include all buckets
                bucket_ids = np.digitize(test_errs, acc_buckets) # satisfies acc_buckets[i-1] <= err < acc_buckets[i]
                nonempty_buckets = np.unique(bucket_ids)
                nonempty_buckets = nonempty_buckets[nonempty_buckets != 0] # remove thetas below error threshold
                for bucket_id in nonempty_buckets:
                    bucket_ix = np.where(bucket_ids == bucket_id)[0]

                    # save model from bucket with lowest loss on clean dataset
                    sv_ix = bucket_ix[np.argmin(train_losses[bucket_ix])]
                    ix_pruned.append(sv_ix)

                thetas = thetas[ix_pruned]
                biases = biases[ix_pruned]


            data_all = {}
            data_all['thetas'] = thetas
            data_all['biases'] = biases
            data_best = {}
            data_best['thetas'] = best_theta
            data_best['biases'] = best_bias

            # save all the target thetas
            if not os.path.isdir('files/target_classifiers/{}/{}/{}'.format(dataset_name,args.model_type, subpop_type)):
                os.makedirs('files/target_classifiers/{}/{}/{}'.format(dataset_name,args.model_type, subpop_type))
            if select_best:
                file_all = open('files/target_classifiers/{}/{}/{}/orig_best_theta_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type, int(subpop_ind),valid_theta_err), 'wb')
                pickle.dump(data_best, file_all,protocol=2)
                file_all.close()
            else:
                file_all = open('files/target_classifiers/{}/{}/{}/orig_thetas_subpop_{}_err-{}'.format(dataset_name,args.model_type, subpop_type, int(subpop_ind),valid_theta_err), 'wb')
                pickle.dump(data_all, file_all,protocol=2)
                file_all.close()
    trn_df.to_csv('files/data/{}_trn_{}_desc.csv'.format(dataset_name, subpop_type), index=False)
