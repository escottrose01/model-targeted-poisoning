from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import hinge_loss
from sklearn import linear_model, svm

from utils import svm_model, logistic_model, calculate_loss, dist_to_boundary, get_subpop_inds, proj_constraint_size, check_boundary_in_constraint_set
import pickle
import argparse
from datasets import load_dataset
import os
import scipy
import pandas as pd
import cvxpy as cp

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
trn_desc_fname = 'files/data/{}_trn_{}_desc.csv'.format(dataset_name, subpop_type)
trn_df = pd.read_csv(trn_desc_fname)

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
            print('Error: invalid loss function passed to CustomLinearModel')
            sys.exit(1)

        if self.penalty == 'l2':
            reg = 0.5 * cp.sum_squares(w)
        else:
            print('Error: invalid penalty function passed to CustomLinearModel')
            sys.exit(1)

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
    clean_loss = hinge_loss(y, y_dec) + 0.5 * Cr * np.inner(model.coef_.T, model.coef_.T)

    constraint_size = max(
        proj_constraint_size(model.coef_, x_lim_tuples[0]),
        proj_constraint_size(model.coef_, x_lim_tuples[1]),
    )

    bound_valid = check_boundary_in_constraint_set(model.coef_, model.intercept_, x_lim_tuples[0]) \
                    or check_boundary_in_constraint_set(model.coef_, model.intercept_, x_lim_tuples[1])

    try:
        if bound_valid:
            model.fit(X, y, force_pos=X_sp)
            y_dec = model.decision_function(X)
            poison_loss = hinge_loss(y, y_dec) + 0.5 * Cr * np.inner(model.coef_.T, model.coef_.T)
            return max(0, X.shape[0] * (poison_loss - clean_loss) / constraint_size)
        else: return 0
    except cp.error.SolverError:
        return 0

def subpop_lower_bound_quantile(X, y, X_sp, Cr, x_lim_tuples, r=1.0):
    clean_model = CustomLinearModel(Cr=Cr, max_iter=int(1e6)).fit(X, y)
    y_dec = clean_model.decision_function(X)
    clean_loss = hinge_loss(y, y_dec) + 0.5 * Cr * np.inner(clean_model.coef_.T, clean_model.coef_.T)

    constraint_size = max(
        proj_constraint_size(clean_model.coef_, x_lim_tuples[0]),
        proj_constraint_size(clean_model.coef_, x_lim_tuples[1]),
    )

    bounds = np.zeros(X_sp.shape[0])
    model = CustomLinearModel(Cr=Cr, max_iter=int(1e6))
    bound_valid = check_boundary_in_constraint_set(clean_model.coef_, clean_model.intercept_, x_lim_tuples[0]) \
                        or check_boundary_in_constraint_set(clean_model.coef_, clean_model.intercept_, x_lim_tuples[1])

    for i, x in enumerate(X_sp):
        try:
            if bound_valid:
                model.fit(X, y, force_pos=x)
                y_dec = model.decision_function(X)
                poison_loss = hinge_loss(y, y_dec) + 0.5 * Cr * np.inner(model.coef_.T, model.coef_.T)
                bounds[i] = max(0, X.shape[0] * (poison_loss - clean_loss) / constraint_size)
            else: bounds[i] = 0
        except cp.error.SolverError:
            bounds[i] = 0

    bounds = np.sort(bounds)
    return bounds[int(np.ceil(r * X_sp.shape[0]) - 1e-6)] # subtract off small delta to get true bound

X_train_cp, y_train_cp = np.copy(X_train), np.copy(y_train)

for kk, subpop_ind in enumerate(subpop_inds):
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

    subpop_lb = subpop_lower_bound(X_train, y_train, tst_sub_x, args.weight_decay, x_lim_tuples)
    subpop_lb_quantile = subpop_lower_bound_quantile(X_train, y_train, tst_sub_x, args.weight_decay, x_lim_tuples, r=valid_theta_err)

    trn_df.loc[subpop_ind, 'Subpop Lower Bound'] = subpop_lb
    trn_df.loc[subpop_ind, 'Subpop Quantile Lower Bound'] = subpop_lb_quantile

trn_df.to_csv(trn_desc_fname, index=False)
