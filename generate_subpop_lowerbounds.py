from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import hinge_loss

from utils import (
    get_subpop_inds, 
    proj_constraint_size, 
    check_boundary_in_constraint_set,
    min_inner_prod,
    max_inner_prod
)
import argparse
from datasets import load_dataset
import os
import sys
import scipy
import pandas as pd
import cvxpy as cp

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or rlogistic regression')
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, 2d_toy, dogfish")
parser.add_argument('--weight_decay',default=0.09, type=float, help='weight decay for regularizers')
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature', 'random'], help='subpopulaton type: cluster, feature, or random')
parser.add_argument('--valid_theta_err', default=None, type=float, help='minimum target model classification error')

args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['adult', 'mnist_17', '2d_toy', 'dogfish', 'loan', 'compas', 'synthetic']

subpop_type = args.subpop_type

valid_theta_err = args.valid_theta_err

X_train, y_train, X_test, y_test = load_dataset(dataset_name)

x_pos_tuple = (0, 1)
x_neg_tuple = (0, 1)
x_lim_tuples = (x_pos_tuple, x_neg_tuple)

if min(y_test) > -1:
    y_test = 2*y_test - 1
if min(y_train) > -1:
    y_train = 2*y_train - 1

# load subpop data
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

# for subpop descriptions
trn_desc_fname = 'files/data/lowerbounds.csv'
num_subpops = max([np.max(v) for v in trn_all_subpops]) + 1
print(num_subpops)
trn_df = pd.DataFrame(index=range(num_subpops), columns=['Subpop Lower Bound', 'Subpop Quantile Lower Bound', 'Subpop Max Lower Bound'])

class CustomLinearModel(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l2', loss='hinge', Cr=1.0, max_iter=10000):
        self.coef_ = None
        self.intercept_ = None
        self.penalty = penalty
        self.loss = loss
        self.max_iter = max_iter
        self.Cr = Cr

    def fit(self, X, y, force_pos=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if min(y) > -1:
            y = 2 * y - 1

        w = cp.Variable(X.shape[1])
        b = cp.Variable()

        margins = X * w + b # * is matrix multuplication

        if self.loss == 'hinge':
            loss = cp.sum_entries(cp.pos(1 - cp.mul_elemwise(y, margins)))
        else:
            raise ValueError('Error: invalid loss function passed to CustomLinearModel')

        if self.penalty == 'l2':
            reg = 0.5 * (cp.sum_squares(w) + b**2)
        else:
            raise ValueError('Error: invalid penalty function passed to CustomLinearModel')

        if force_pos is not None:
            EPS = 1e-12
            force_margins = force_pos * w + b # * is matrix multuplication
            constraints = [force_margins >= EPS]
        else:
            constraints = []

        objective = cp.Minimize(loss / X.shape[0] + self.Cr * reg)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI, max_iter=self.max_iter)

        self.coef_ = w.value
        self.intercept_ = b.value

        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')

        X = check_array(X)

        margins = np.dot(X,self.coef_) + self.intercept_
        return np.where(margins < 0, 0, 1)

    def decision_function(self, X):
        check_is_fitted(self, 'coef_')

        X = check_array(X)

        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        check_is_fitted(self, 'coef_')

        X = check_array(X)

        margins = np.dot(X, self.coef_) + self.intercept_
        y_pred = np.where(margins < 0, 0, 1)
        return np.sum(y_pred == y) / X.shape[0]


def subpop_lower_bound(X, y, X_sp, Cr, x_lim_tuples):
    # assumption: decision boundary lies within constraint set
    # this assumption may fail when model is overregularized
    # or when classification task is not well-posed

    model = CustomLinearModel(Cr=Cr, max_iter=int(1e6)).fit(X, y)
    y_dec = model.decision_function(X)
    reg = 0.5 * Cr * (np.inner(model.coef_.T, model.coef_.T) + model.intercept_ * model.intercept_)
    clean_loss = hinge_loss(y, y_dec) + reg

    # previously, we used the constraint size
    # constraint_size = max(
    #     proj_constraint_size(model.coef_, x_lim_tuples[0]),
    #     proj_constraint_size(model.coef_, x_lim_tuples[1]),
    # )

    # now, we compute the max loss point directly
    min_margin = min(
        model.intercept_ + min_inner_prod(model.coef_, x_lim_tuples[0]), # y=1
        model.intercept_ - max_inner_prod(model.coef_, x_lim_tuples[1]), # y=-1
    )
    sup_loss = max(0.0, 1.0 - min_margin) # note: can replace with any nondecreasing margin-based loss
    sup_reg_loss = sup_loss + reg

    # bound_valid = check_boundary_in_constraint_set(model.coef_, model.intercept_, x_lim_tuples[0]) \
    #                 or check_boundary_in_constraint_set(model.coef_, model.intercept_, x_lim_tuples[1])
    bound_valid = True

    try:
        if bound_valid:
            model.fit(X, y, force_pos=X_sp)
            y_dec = model.decision_function(X)
            reg = 0.5 * Cr * (np.inner(model.coef_.T, model.coef_.T) + model.intercept_ * model.intercept_)
            poison_loss = hinge_loss(y, y_dec) + reg
            # return max(0, X.shape[0] * (poison_loss - clean_loss) / (1.0 + constraint_size))
            return max(0, X.shape[0] * (poison_loss - clean_loss) / sup_reg_loss)
        else: return 0
    except cp.error.SolverError:
        return 0

X_train_cp, y_train_cp = np.copy(X_train), np.copy(y_train)

# get score for each individual test datapoint
indiv_scores = np.zeros(X_test.shape[0])
subpop_scores = np.zeros(num_subpops)

indiv_jobs = []
indiv_job_ix = []
for i in range(X_test.shape[0]):
    if y_test[i] == 1:
        # we ignore positive-labeled points in subpops
        continue

    X_sp = X_test[i]
    indiv_jobs.append(delayed(subpop_lower_bound)(X_train_cp, y_train_cp, X_sp, args.weight_decay, x_lim_tuples))
    indiv_job_ix.append(i)

subpop_jobs = []
for i, subpop_ind in enumerate(subpop_inds):
    tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
    trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
    # indices of points belong to subpop
    tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, y_test, y_train)

    # get the corresponding points in the dataset
    tst_sub_x, tst_sub_y = X_test[tst_sbcl], y_test[tst_sbcl]
    tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], y_test[tst_non_sbcl]
    trn_sub_x, trn_sub_y = X_train_cp[trn_sbcl], y_train_cp[trn_sbcl]
    trn_nsub_x, trn_nsub_y = X_train_cp[trn_non_sbcl], y_train_cp[trn_non_sbcl]

    subpop_jobs.append(delayed(subpop_lower_bound)(X_train_cp, y_train_cp, tst_sub_x, args.weight_decay, x_lim_tuples))

print(len(indiv_jobs), len(subpop_jobs), subpop_inds)
for subpop_ind, score in zip(subpop_inds, Parallel(n_jobs=-1)(subpop_jobs)):
    subpop_scores[int(subpop_ind)] = score

for i, score in zip(indiv_job_ix, Parallel(n_jobs=-1)(indiv_jobs)):
    indiv_scores[i] = score
print("most interesting bound: ", max(indiv_scores))
print("most interesting subpop bound: ", max(subpop_scores))

# get score for each subpopulation
for _, subpop_ind in enumerate(subpop_inds):
    tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
    trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
    # indices of points belong to subpop
    tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, y_test, y_train)

    subpop_bounds = np.sort(indiv_scores[tst_sbcl])
    subpop_lb_quantile = subpop_bounds[int(np.ceil(valid_theta_err * tst_sbcl[0].shape[0]) - 1e-6)] # subtract off small delta to get true bound
    subpop_lb = subpop_scores[int(subpop_ind)]

    trn_df.loc[subpop_ind, 'Subpop Lower Bound'] = subpop_lb
    trn_df.loc[subpop_ind, 'Subpop Quantile Lower Bound'] = subpop_lb_quantile
    trn_df.loc[subpop_ind, 'Subpop Max Lower Bound'] = max(subpop_bounds)

trn_df.to_csv(trn_desc_fname, index=False)
