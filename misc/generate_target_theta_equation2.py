from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn import linear_model, svm
# from utils import *
from utils import svm_model, calculate_loss, dist_to_boundary, cvx_dot, get_subpop_inds
import cvxpy as cvx

import pickle
import argparse
from datasets import load_dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature'], help='subpopulaton type: cluster (original subpops in paper), mixed (subpops from paper mixed positive and negative), or feature')
parser.add_argument('--alpha_value', default=.1, type=float, help='alpha value to use in optimization equation')
parser.add_argument('--mixed', default = False, type=bool, help='True (has both positive and negative points) or False (has only negative points)')
args = parser.parse_args()
subpop_type = args.subpop_type
alpha_value = args.alpha_value
mixed = args.mixed

# for now, will only be trying to do subpopulation on adult
dataset_name = 'adult'
subpop = True

# load data
X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)
if min(Y_test)>-1:
    Y_test = 2*Y_test-1
if min(Y_train) > -1:
    Y_train = 2*Y_train - 1

print(np.amax(Y_train),np.amin(Y_train))
weight_decay = .09
#weight_decay = 1e-5

max_iter = -1
fit_intercept = True

ScikitModel = svm_model
C = 1.0 / (X_train.shape[0] * weight_decay)
model = ScikitModel(
    C=C,
    tol=1e-10,
    fit_intercept=fit_intercept,
    random_state=24,
    verbose=False,
    max_iter = 1000)
model.fit(X_train, Y_train)

model_dumb = ScikitModel(
    C=C,
    tol=1e-10,
    fit_intercept=fit_intercept,
    random_state=24,
    verbose=False,
    max_iter = 1000)
model_dumb.fit(X_train, Y_train)

orig_theta = model.coef_.reshape(-1)
orig_bias = model.intercept_[0]
norm = np.sqrt(np.linalg.norm(orig_theta)**2 + orig_bias**2)
print("norm of clean model:",norm)
# calculate the clean model acc
train_acc = model.score(X_train,Y_train)
test_acc = model.score(X_test,Y_test)

print(orig_theta.shape,X_train.shape,orig_bias.shape,Y_train.shape)
margins = Y_train*(X_train.dot(orig_theta) + orig_bias)
train_loss, train_err = calculate_loss(margins)
reg = (weight_decay/2) * np.linalg.norm(orig_theta)**2
print("train_acc:{}, train loss:{}, train error:{}".format(train_acc,train_loss+reg,train_err))
# test margins and loss
margins = Y_test*(X_test.dot(orig_theta) + orig_bias)
test_loss, test_err = calculate_loss(margins)
print("test_acc:{}, test loss:{}, test error:{}".format(test_acc,test_loss+reg,test_err))

# we prefer points with lower loss (higher loss in correct labels)
margins = Y_train*(X_train.dot(orig_theta) + orig_bias)

class search_target_theta(object):
    def __init__(self,D_c,D_sub):
        # define the bias and variable terms
        X_train, y_train = D_c
        x_sub, y_sub = D_sub

        d = X_train.shape[1] # dimension of theta

        self.cvx_theta_p = cvx.Variable(d)
        self.cvx_bias_p = cvx.Variable()

        #ALPHA VALUE
        self.alpha = cvx.Parameter(sign='positive')
#        self.alpha = .1

        reg = cvx.pnorm(self.cvx_theta_p, 2)**2
        self.L = cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(y_train, X_train * self.cvx_theta_p + self.cvx_bias_p)))/X_train.shape[0] + (weight_decay/2) * reg
        self.M = cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(y_sub, x_sub * self.cvx_theta_p + self.cvx_bias_p)))/x_sub.shape[0] + + (weight_decay/2) * reg

        self.equation_two = self.L + (self.alpha * self.M)

        self.cvx_objective = cvx.Minimize(self.equation_two)
        self.cvx_prob = cvx.Problem(self.cvx_objective)

    def solve(self,
            alpha,
            verbose = False
            ):

        self.alpha.value = alpha

        self.cvx_prob.solve(verbose=verbose, solver=cvx.GUROBI)

        target_theta = np.array(self.cvx_theta_p.value)
        target_theta = target_theta.reshape(-1)
        target_bias = np.array(self.cvx_bias_p.value)

        return target_theta, target_bias

y_list = [1,-1]

# do the clustering and attack each subpopulation
# generation process for subpop: directly flip the labels of subpop
# choose based on arguments
from sklearn import cluster
num_clusters = 20
pois_rates = [0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5]

# find the clusters and corresponding subpop size

##cls_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
##if os.path.isfile(cls_fname):
##    trn_km = np.loadtxt(cls_fname)
##    cls_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
##    tst_km = np.loadtxt(cls_fname)
##else:
##    if(subpop_type=='cluster' or subpop_type=='mixed'):
##        num_clusters = 20
##        km = cluster.KMeans(n_clusters=num_clusters,random_state = 0)
##        km.fit(X_train)
##        trn_km = km.labels_
##        tst_km = km.predict(X_test)
##        # save the clustering info to ensure everything is reproducible
##        cls_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
##        np.savetxt(cls_fname,trn_km)
##        cls_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
##        np.savetxt(cls_fname,tst_km)
##    elif(subpop_type=='feature'):
##        print('AHHHHH!')
##
##cl_inds, cl_cts = np.unique(trn_km, return_counts=True)
##
##print(cl_inds, cl_cts)

trn_subpop_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
with open(trn_subpop_fname, 'r') as f:
    trn_all_subpops = [np.array(map(int, line.split())) for line in f]
    
tst_subpop_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
with open(tst_subpop_fname, 'r') as f:
    tst_all_subpops = [np.array(map(int, line.split())) for line in f]

subpops_flattened = np.concatenate(trn_all_subpops).flatten()
subpop_inds, subpop_cts = np.unique(subpops_flattened, return_counts=True)

if(not mixed):
    trn_sub_accs = []
    for i in range(len(subpop_cts)):
        subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
        print("subpop ID and Size:",subpop_ind,subpop_ct)
        # indices of points belong to subpop
        tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
        trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
        tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train)

        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
        tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
        trn_sub_acc = model.score(trn_sub_x, trn_sub_y)
        # check the target and collateral damage info
        print("----------Subpop Indx: {} ------".format(subpop_ind))
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print('Clean Test Target Acc : %.3f' % tst_sub_acc)
        print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
        trn_sub_accs.append(trn_sub_acc)

    print(subpop_inds, subpop_cts)

    # print(tst_sub_accs)
    # sort the subpop based on tst acc and choose 5 highest ones
    highest_5_inds = np.argsort(trn_sub_accs)[-3:]
    #highest_5_inds = np.argsort(tst_sub_accs)[-5:]
    subpop_inds = subpop_inds[highest_5_inds]
    subpop_cts = subpop_cts[highest_5_inds]
    print(subpop_inds, subpop_cts)
else:
    trn_sub_succ = []
    for i in range(len(subpop_cts)):
        subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
        print("subpop ID and Size:",subpop_ind,subpop_ct)
        # indices of points belong to subpop
        tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
        trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])

        tst_sbcl = np.where(tst_subpop_inds)
        trn_sbcl = np.where(trn_subpop_inds)
        tst_non_sbcl = np.where(np.logical_not(tst_subpop_inds))
        trn_non_sbcl = np.where(np.logical_not(trn_subpop_inds))

        tst_neg_sbcl = np.where(np.logical_and(tst_subpop_inds, Y_test == -1))[0]
        trn_neg_sbcl = np.where(np.logical_and(trn_subpop_inds, Y_train == -1))[0]
        tst_pos_sbcl = np.where(np.logical_and(tst_subpop_inds, Y_test == 1))[0]
        trn_pos_sbcl = np.where(np.logical_and(trn_subpop_inds, Y_train == 1))[0]
        
        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
        tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
        trn_sub_acc = model.score(trn_sub_x, trn_sub_y)
        # check the target and collateral damage info
        print("----------Subpop Indx: {} ------".format(subpop_ind))
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print('Clean Test Target Acc : %.3f' % tst_sub_acc)
        print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
        pos_trn_sub_y = Y_train[trn_pos_sbcl]
        neg_trn_sub_y = Y_train[trn_neg_sbcl]
        flipped_trn_sub_y = np.append(pos_trn_sub_y,-neg_trn_sub_y)
        print('Percentage of training points being classified as positive : %.3f' % model.score(trn_sub_x,flipped_trn_sub_y))
        trn_sub_succ.append(model.score(trn_sub_x,flipped_trn_sub_y))

    #Only want ones that are below 50% success already
    print(subpop_inds, subpop_cts)
    how_many_below = 0
    for x in trn_sub_succ:
        if(x < .5):
            how_many_below = how_many_below + 1
    below_50 = np.argsort(trn_sub_succ)[0:(how_many_below)]
    subpop_inds = subpop_inds[below_50]
    subpop_cts = subpop_cts[below_50]
    print(subpop_inds, subpop_cts)
    

# save the selected subpop info
cls_fname = 'files/data/{}_svm_{}_selected_subpops.txt'.format(dataset_name, subpop_type)
np.savetxt(cls_fname,np.array([subpop_inds,subpop_cts]))
print("#---------Selected Subpops------#")
for i in range(len(subpop_cts)):
    subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
    print("cluster ID and Size:",subpop_ind,subpop_ct)
    # indices of points belong to subpop
    tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
    trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])

    if(not mixed):
        tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train)
        # make sure subpop is from class -1
        assert (tst_sub_y == -1).all()
        assert (trn_sub_y == -1).all()
    else:
        tst_sbcl = np.where(tst_subpop_inds)
        trn_sbcl = np.where(trn_subpop_inds)
        tst_non_sbcl = np.where(np.logical_not(tst_subpop_inds))
        trn_non_sbcl = np.where(np.logical_not(trn_subpop_inds))

        tst_neg_sbcl = np.where(np.logical_and(tst_subpop_inds, Y_test == -1))[0]
        trn_neg_sbcl = np.where(np.logical_and(trn_subpop_inds, Y_train == -1))[0]
        tst_pos_sbcl = np.where(np.logical_and(tst_subpop_inds, Y_test == 1))[0]
        trn_pos_sbcl = np.where(np.logical_and(trn_subpop_inds, Y_train == 1))[0]
    
    # get the corresponding points in the dataset
    tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
    tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
    trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
    trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]
    tst_sub_acc = model.score(tst_sub_x, tst_sub_y)

    # check the target and collateral damage info
    print("----------Subpop Indx: {}------".format(subpop_ind))
    print('Clean Total Train Acc : %.3f' % model.score(X_train, Y_train))
    print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
    print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
    print('Clean Total Test Acc : %.3f' % model.score(X_test, Y_test))
    print('Clean Test Target Acc : %.3f' % tst_sub_acc)
    print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
    print()
    if(mixed):
        print('Number of NEGATIVE TRAIN POINTS : ', trn_neg_sbcl.size)
        print('Number of POSITIVE TRAIN POINTS : ', trn_pos_sbcl.size)
        print('Number of TOTAL TRAIN POINTS : ', trn_neg_sbcl.size + trn_pos_sbcl.size)
        print('Number of negative test points : ', tst_neg_sbcl.size)
        print('Number of positive test points : ', tst_pos_sbcl.size)
        print('Number of total test points : ', tst_neg_sbcl.size + tst_pos_sbcl.size)
        print()
        print('Accuracy on negative train points : %.3f' % model.score(X_train[trn_neg_sbcl],Y_train[trn_neg_sbcl]))
        print('Accuracy on positive train points : %.3f' % model.score(X_train[trn_pos_sbcl],Y_train[trn_pos_sbcl]))
        print('Accuracy on negative test points : %.3f' % model.score(X_test[tst_neg_sbcl],Y_test[tst_neg_sbcl]))
        print('Accuracy on positive test points : %.3f' % model.score(X_test[tst_pos_sbcl],Y_test[tst_pos_sbcl]))
        print()
        pos_trn_sub_y = Y_train[trn_pos_sbcl]
        neg_trn_sub_y = Y_train[trn_neg_sbcl]
        flipped_trn_sub_y = np.append(pos_trn_sub_y,-neg_trn_sub_y)
        print('Percentage of training points being classified as positive : %.3f' % model.score(trn_sub_x,flipped_trn_sub_y))
        assert(flipped_trn_sub_y == 1).all()

    # solve the optimization problemand obtain the target classifier,
    if(not mixed):
        find_target_theta = search_target_theta((X_train,Y_train),(trn_sub_x,-trn_sub_y))
    else:
        find_target_theta = search_target_theta((X_train,Y_train),(trn_sub_x,flipped_trn_sub_y))

    target_theta, target_bias = find_target_theta.solve(alpha_value, verbose = False)
    norm = np.sqrt(np.linalg.norm(target_theta)**2 + target_bias**2)
    print("norm of target theta:",norm)

    model_dumb.coef_ = np.array([target_theta])
    model_dumb.intercept_ = np.array([target_bias])
    # print out the acc info on subpop
    trn_sub_acc = model_dumb.score(trn_sub_x, trn_sub_y)
    # print("----------Subpop Indx: {}------".format(subpop_ind))
    print('Target Total Train Acc : %.3f' % model_dumb.score(X_train, Y_train))
    print('Target Train Target Acc : %.3f' % model_dumb.score(trn_sub_x, trn_sub_y))
    print('Target Train Collat Acc : %.3f' % model_dumb.score(trn_nsub_x,trn_nsub_y))
    print('Target Total Test Acc : %.3f' % model_dumb.score(X_test, Y_test))
    print('Target Test Target Acc : %.3f' % model_dumb.score(tst_sub_x, tst_sub_y))
    print('Target Test Collat Acc : %.3f' % model_dumb.score(tst_nsub_x, tst_nsub_y))
    print()
    if(mixed):
        print('Accuracy on negative train points : %.3f' % model_dumb.score(X_train[trn_neg_sbcl],Y_train[trn_neg_sbcl]))
        print('Accuracy on positive train points : %.3f' % model_dumb.score(X_train[trn_pos_sbcl],Y_train[trn_pos_sbcl]))
        print('Accuracy on negative test points : %.3f' % model_dumb.score(X_test[tst_neg_sbcl],Y_test[tst_neg_sbcl]))
        print('Accuracy on positive test points : %.3f' % model_dumb.score(X_test[tst_pos_sbcl],Y_test[tst_pos_sbcl]))
        print()
        print('Percentage of training points being classified as positive : %.3f' % model_dumb.score(trn_sub_x,flipped_trn_sub_y))


    data_all = {}
    data_all['thetas'] = target_theta
    data_all['biases'] = target_bias
    # save the target thetas
    if not os.path.isdir('files/target_classifiers/{}/svm/{}'.format(dataset_name, subpop_type)):
        os.makedirs('files/target_classifiers/{}/svm/{}'.format(dataset_name, subpop_type))
    file_all = open('files/target_classifiers/{}/svm/{}/orig_best_theta_subpop_{}_err-1.0'.format(dataset_name, subpop_type, int(subpop_ind)), 'wb')
    pickle.dump(data_all, file_all,protocol=2)
    file_all.close()
