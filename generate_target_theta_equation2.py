from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn import linear_model, svm
# from utils import *
from utils import svm_model, calculate_loss, dist_to_boundary, cvx_dot
import cvxpy as cvx

import pickle
import argparse
from datasets import load_dataset
import os

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
# choose 5 with highest original acc
from sklearn import cluster
num_clusters = 20
pois_rates = [0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5]

cls_fname = 'files/data/{}_trn_cluster_labels.txt'.format(dataset_name)
if os.path.isfile(cls_fname):
    trn_km = np.loadtxt(cls_fname)
    cls_fname = 'files/data/{}_tst_cluster_labels.txt'.format(dataset_name)
    tst_km = np.loadtxt(cls_fname)
else:
    num_clusters = 20
    km = cluster.KMeans(n_clusters=num_clusters,random_state = 0)
    km.fit(X_train)
    trn_km = km.labels_
    tst_km = km.predict(X_test)
    # save the clustering info to ensure everything is reproducible
    cls_fname = 'files/data/{}_trn_cluster_labels.txt'.format(dataset_name)
    np.savetxt(cls_fname,trn_km)
    cls_fname = 'files/data/{}_tst_cluster_labels.txt'.format(dataset_name)
    np.savetxt(cls_fname,tst_km)
# find the clusters and corresponding subpop size
cl_inds, cl_cts = np.unique(trn_km, return_counts=True)
trn_sub_accs = []
for i in range(len(cl_cts)):
    cl_ind, cl_ct = cl_inds[i], cl_cts[i]
    print("cluster ID and Size:",cl_ind,cl_ct)         
    # indices of points belong to cluster
    tst_sbcl = np.where(np.logical_and(tst_km==cl_ind,Y_test == -1))
    trn_sbcl = np.where(np.logical_and(trn_km==cl_ind,Y_train == -1))
    tst_non_sbcl = np.where(np.logical_or(tst_km!=cl_ind,Y_test != -1))
    trn_non_sbcl = np.where(np.logical_or(trn_km!=cl_ind,Y_train != -1))
    
    # get the corresponding points in the dataset
    tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
    tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
    trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
    trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]  
    tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
    trn_sub_acc = model.score(trn_sub_x, trn_sub_y)
    # check the target and collateral damage info
    print("----------Subpop Indx: {} ------".format(cl_ind))
    print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
    print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
    print('Clean Test Target Acc : %.3f' % tst_sub_acc)
    print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
    trn_sub_accs.append(trn_sub_acc)

print(cl_inds, cl_cts)
# print(tst_sub_accs)
# sort the subpop based on tst acc and choose 5 highest ones
highest_5_inds = np.argsort(trn_sub_accs)[-3:]
#highest_5_inds = np.argsort(tst_sub_accs)[-5:]
cl_inds = cl_inds[highest_5_inds]
cl_cts = cl_cts[highest_5_inds]
print(cl_inds, cl_cts)

# save the selected subpop info
cls_fname = 'files/data/{}_svm_selected_subpops.txt'.format(dataset_name)
np.savetxt(cls_fname,np.array([cl_inds,cl_cts]))
print("#---------Selected Subpops------#")
for i in range(len(cl_cts)): 
    cl_ind, cl_ct = cl_inds[i], cl_cts[i]
    print("cluster ID and Size:",cl_ind,cl_ct)
    # indices of points belong to cluster
    tst_sbcl = np.where(np.logical_and(tst_km==cl_ind,Y_test == -1))[0]
    trn_sbcl = np.where(np.logical_and(trn_km==cl_ind,Y_train == -1))[0]
    tst_non_sbcl = np.where(np.logical_or(tst_km!=cl_ind,Y_test != -1))[0]
    trn_non_sbcl = np.where(np.logical_or(trn_km!=cl_ind,Y_train != -1))[0]
    
    # get the corresponding points in the dataset
    tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
    tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
    trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
    trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]  
    tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
    # make sure subpop is from class -1
    assert (tst_sub_y == -1).all()
    assert (trn_sub_y == -1).all()
    # check the target and collateral damage info
    print("----------Subpop Indx: {}------".format(cl_ind))
    print('Clean Total Train Acc : %.3f' % model.score(X_train, Y_train))
    print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
    print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
    print('Clean Total Test Acc : %.3f' % model.score(X_test, Y_test))
    print('Clean Test Target Acc : %.3f' % tst_sub_acc)
    print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))

    # solve the optimization problemand obtain the target classifier,
    find_target_theta = search_target_theta((X_train,Y_train),(trn_sub_x,-trn_sub_y))

    target_theta, target_bias = find_target_theta.solve(100., verbose = False)
    norm = np.sqrt(np.linalg.norm(target_theta)**2 + target_bias**2)
    print("norm of target theta:",norm)

    model_dumb.coef_ = np.array([target_theta])
    model_dumb.intercept_ = np.array([target_bias])
    # print out the acc info on subpop
    trn_sub_acc = model_dumb.score(trn_sub_x, trn_sub_y)
    # print("----------Subpop Indx: {}------".format(cl_ind))
    print('Target Total Train Acc : %.3f' % model_dumb.score(X_train, Y_train))
    print('Target Train Target Acc : %.3f' % model_dumb.score(trn_sub_x, trn_sub_y))
    print('Target Train Collat Acc : %.3f' % model_dumb.score(trn_nsub_x,trn_nsub_y))
    print('Target Total Test Acc : %.3f' % model_dumb.score(X_test, Y_test))
    print('Target Test Target Acc : %.3f' % model_dumb.score(tst_sub_x, tst_sub_y))
    print('Target Test Collat Acc : %.3f' % model_dumb.score(tst_nsub_x, tst_nsub_y))
    print('ACTUAL THETA : ')
    print(target_theta)

    data_all = {}
    data_all['thetas'] = target_theta
    data_all['biases'] = target_bias      
    # save the target thetas
    if not os.path.isdir('files/target_classifiers/{}/svm'.format(dataset_name)):
        os.makedirs('files/target_classifiers/{}/svm'.format(dataset_name))
    file_all = open('files/target_classifiers/{}/svm/orig_best_theta_subpop_{}_err-1.0'.format(dataset_name,int(cl_ind)), 'wb')
    pickle.dump(data_all, file_all,protocol=2)
    file_all.close()
