from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import argparse
import os
import csv
from sklearn import cluster
import sklearn
import pandas as pd
from scipy.spatial.distance import pdist, cdist

from datasets import load_dataset, load_dataset_cols, load_dataset_feature_inds
from utils import get_subpop_inds

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='adult', help='four datasets: mnist_17, adult, 2d_toy, dogfish')
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature', 'random'], help='subpopulaton type: cluster, feature, or random')
parser.add_argument('--subpop_ratio', default='0.05', help='desired subpopulation ratio for feature subpop: between 0 and 1; or <0 to ignore')
parser.add_argument('--tolerance', default=0.001, type=float, help='tolerance value for subpop_ratio')
parser.add_argument('--num_subpops', default='auto', help='number of clusters (subpops) to use for clustering (or random) subpop: integer or \'auto\' for default')

args = parser.parse_args()



NUM_SUBPOPS = {
    'dogfish' : 3,
    'mnist_17' : 20,
    'adult' : 20,
    '2d_toy' : 4,
    'loan' : 20,
    'compas' : 20
}

assert args.dataset in ['dogfish', 'mnist_17', 'adult', '2d_toy', 'loan', 'compas'], 'not a valid dataset!'
dataset_name = args.dataset

subpop_type = args.subpop_type

assert (args.num_subpops == 'auto') or (args.num_subpops.isdigit()), 'not a valid num_subpops argument!'
num_subpops = NUM_SUBPOPS[dataset_name] if args.num_subpops == 'auto' else int(args.num_subpops)

try:
    subpop_ratio = float(args.subpop_ratio)
except ValueError:
    assert False, 'not a valid subpop_ratio argument!'

generate_tst_desc = False



subpop_fname = 'files/data/{}_{}_labels.txt'.format(dataset_name, subpop_type)
if os.path.isfile(subpop_fname):
    sys.exit(0)

X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)

if min(Y_test) > -1:
    Y_test = 2*Y_test - 1
if min(Y_train) > -1:
    Y_train = 2*Y_train - 1

print('instance in train and test data: {}, {}'.format(X_train.shape[0], X_test.shape[0]))

if subpop_type == 'cluster':
    seed = 0
    ready = False
    while not ready:
        km = cluster.KMeans(n_clusters=num_subpops, random_state=seed)
        km.fit(X_train)
        trn_km = km.labels_
        tst_km = km.predict(X_test)

        ready = True
        for subpop_ix in range(num_subpops):
            if np.sum(Y_train[trn_km == subpop_ix] == -1) < 1 \
                or np.sum(Y_train[trn_km == subpop_ix] == 1) < 1\
                or np.sum(Y_test[tst_km == subpop_ix] == -1) < 1 \
                or np.sum(Y_test[tst_km == subpop_ix] == 1) < 1:
                ready = False
                break

        print(seed, ready)
        seed += 1

    trn_all_subpops = [[x] for x in list(trn_km)]
    tst_all_subpops = [[x] for x in list(tst_km)]
elif subpop_type == 'feature':
    assert dataset_name in ['adult', 'loan', 'compas'], 'feature matching not supported for this dataset!'
    all_cols = load_dataset_cols(dataset_name)
    feature_inds, feature_descriptions = load_dataset_feature_inds(dataset_name)
    descs = []

    trn_all_subpops = [[] for i in range(X_train.shape[0])]
    tst_all_subpops = [[] for i in range(X_test.shape[0])]
    subcl_ix = 0

    # we generate subpopulations by feature matching subsets of features
    feature_subsets = []
    for i in range(1, 1 << len(feature_inds)):
        feature_subsets.append([j for j in range(len(feature_inds)) if (1 << j)&i])

    for feature_subset in feature_subsets:
        feature_names = [feature_descriptions[i] for i in feature_subset]

        subcl_inds_nested = [feature_inds[i] for i in feature_subset]
        if (len(subcl_inds_nested) > 3): continue # might be worth parameterizing the feature number threshhold
        subcl_inds = [item for sublist in subcl_inds_nested for item in sublist]
        trn_prot = X_train[:, subcl_inds]
        tst_prot = X_test[:, subcl_inds]
        prot_cols = [all_cols[i] for i in subcl_inds]
        subclasses, counts = np.unique(trn_prot, axis=0, return_counts=True) # use only points filtered on fixed features for counts

        if subpop_ratio >= 0.:
            count_min = (subpop_ratio - args.tolerance) * X_train.shape[0]
            count_max = (subpop_ratio + args.tolerance) * X_train.shape[0]
        else:
            count_min, count_max = 0, X_train.shape[0] + 1

        for subcl, count in zip(subclasses, counts):
            if (count_min < count < count_max):
                trn_subcl = np.where(np.linalg.norm(trn_prot - subcl, axis=1) == 0)[0]
                tst_subcl = np.where(np.linalg.norm(tst_prot - subcl, axis=1) == 0)[0]

                # make sure the subpop contains instances from both classes
                # in both the train and test set
                if np.sum(Y_train[trn_subcl] == -1) < 1 \
                    or np.sum(Y_train[trn_subcl] == 1) < 1 \
                    or np.sum(Y_test[tst_subcl] == -1) < 1 \
                    or np.sum(Y_test[tst_subcl] == 1) < 1:
                    print('omitting subpop {}'.format(subcl_ix))
                    continue

                for ix in trn_subcl:
                    trn_all_subpops[ix].append(subcl_ix)
                for ix in tst_subcl:
                    tst_all_subpops[ix].append(subcl_ix)
                subcl_ix += 1

                desc = {}
                base_ix = 0
                for feature_name, inds in zip(feature_names, subcl_inds_nested):
                    # one_hot_encoded = [subcl[i] for i in inds]#all_cols[feature_inds]
                    one_hot_encoded = subcl[base_ix:base_ix+len(inds)]
                    base_ix += len(inds)
                    hot_ix = np.where(one_hot_encoded==1)
                    if (hot_ix[0].shape[0] == 0):
                        col_name = 'Unknown'
                    else:
                        col_name = all_cols[hot_ix[0][0] + inds[0]]
                    desc[str(feature_name)] = str(col_name)

                print('comparing', feature_names, ', found match filtering on')
                print('\t', [prot_col for v, prot_col in zip(subcl, prot_cols) if v > 0.5], count)
                print('\t% of train data:', trn_subcl.shape[0]/X_train.shape[0])
                print('\t% of test data: ', tst_subcl.shape[0]/X_test.shape[0])
                descs.append(desc)
elif subpop_type == 'random':
    print('generating {} random subpops'.format(num_subpops))
    trn_all_subpops = [[] for i in range(X_train.shape[0])]
    tst_all_subpops = [[] for i in range(X_test.shape[0])]
    np.random.seed(num_subpops + int(10000000 * subpop_ratio)) # want same subpops for same params
    for subpop_ind in range(num_subpops):
        num_trn_samples = int(subpop_ratio * X_train.shape[0])
        num_tst_samples = int(subpop_ratio * X_test.shape[0])
        trn_subcl = np.random.permutation(X_train.shape[0])[:num_trn_samples]
        tst_subcl = np.random.permutation(X_test.shape[0])[:num_tst_samples]

        for ix in trn_subcl:
            trn_all_subpops[ix].append(subpop_ind)
        for ix in tst_subcl:
            tst_all_subpops[ix].append(subpop_ind)


# save the subpop info to ensure everything is reproducible
delim = str(' ').encode('utf-8') # convert to utf-8
trn_subpop_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
with open(trn_subpop_fname, 'w') as f:
    csv.writer(f, delimiter=delim).writerows(trn_all_subpops)
tst_subpop_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
with open(tst_subpop_fname, 'w') as f:
    csv.writer(f, delimiter=delim).writerows(tst_all_subpops)

print('Generated and saved {} subpops!'.format(1 + max([max(l) for l in trn_all_subpops if len(l) > 0])))

# after generating the subpops, we additionally produce information describing them
with open(trn_subpop_fname, 'r') as f:
    trn_all_subpops = [np.array(map(int, line.split())) for line in f]
with open(tst_subpop_fname, 'r') as f:
    tst_all_subpops = [np.array(map(int, line.split())) for line in f]

subpops_flattened = np.concatenate(trn_all_subpops).flatten()
subpop_inds, subpop_cts = np.unique(subpops_flattened, return_counts=True)

# columns = ['Total Pts', 'Subpop Pts', 'Num Positive', 'Num Negative', 'Type', 'Gini Impurity', 'Avg Silhouette', 'Davies Bouldin', 'Avg Distance Ratio', 'Linear Sep Score']
trn_df = pd.DataFrame()
tst_df = pd.DataFrame()
for i in range(len(subpop_cts)):
    subpop_ind, subpop_ct = subpop_inds[i], subpop_cts[i]
    print("subpop ID and Size:", subpop_ind, subpop_ct)
    # indices of points belong to subpop
    tst_subpop_inds = np.array([np.any(v == subpop_ind) for v in tst_all_subpops])
    trn_subpop_inds = np.array([np.any(v == subpop_ind) for v in trn_all_subpops])
    tst_sbcl, trn_sbcl, tst_non_sbcl, trn_non_sbcl = get_subpop_inds(dataset_name, tst_subpop_inds, trn_subpop_inds, Y_test, Y_train, mixed=True)

    # get the corresponding points in the dataset
    tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
    tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
    trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
    trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]

    # these are NOT cluster labels used in clustering subpop!
    # always two "clusters", either subpop or !subpop
    trn_dummy_cluster_labels = np.where(trn_subpop_inds, np.zeros(X_train.shape[0]), np.ones(X_train.shape[0]))
    tst_dummy_cluster_labels = np.where(tst_subpop_inds, np.zeros(X_test.shape[0]), np.ones(X_test.shape[0]))
    trn_dummy_svm_labels = 2 * trn_dummy_cluster_labels - 1
    tst_dummy_svm_labels = 2 * tst_dummy_cluster_labels - 1

    trn_frac_pos = np.sum(trn_sub_y == 1) / trn_sub_x.shape[0]
    tst_frac_pos = np.sum(trn_sub_y == 1) / tst_sub_x.shape[0]

    trn_silhouettes = sklearn.metrics.silhouette_samples(X_train, trn_dummy_cluster_labels)
    trn_df.loc[i, 'Total Pts'] = X_train.shape[0]
    trn_df.loc[i, 'Subpop Pts'] = trn_sub_x.shape[0]
    trn_df.loc[i, 'Num Positive'] = np.sum(trn_sub_y == 1)
    trn_df.loc[i, 'Num Negative'] = np.sum(trn_sub_y == -1)
    trn_df.loc[i, 'Type'] = subpop_type
    trn_df.loc[i, 'Gini Impurity'] = 1. - (trn_frac_pos**2 + (1. - trn_frac_pos)**2)
    trn_df.loc[i, 'Avg Silhouette'] = np.mean(trn_silhouettes[trn_subpop_inds])
    trn_df.loc[i, 'Davies Bouldin'] = sklearn.metrics.davies_bouldin_score(X_train, trn_dummy_cluster_labels)
    trn_df.loc[i, 'Avg Distance Ratio'] = np.mean(pdist(trn_sub_x)) / np.mean(cdist(trn_sub_x, trn_nsub_x))
    trn_df.loc[i, 'Linear Sep Score'] = sklearn.svm.LinearSVC(C=10000.0).fit(X_train, trn_dummy_svm_labels).score(trn_sub_x, trn_dummy_svm_labels[trn_sbcl]) # score assuming |subpop| << |all data|
    if subpop_type == 'feature':
        trn_df.loc[i, 'Semantic Info'] = str(descs[i]).replace("'", '"')

    if generate_tst_desc:
        tst_silhouettes = sklearn.metrics.silhouette_samples(X_test, tst_dummy_cluster_labels)
        tst_df.loc[i, 'Total Pts'] = X_test.shape[0]
        tst_df.loc[i, 'Subpop Pts'] = tst_sub_x.shape[0]
        tst_df.loc[i, 'Num Positive'] = np.sum(tst_sub_y == 1)
        tst_df.loc[i, 'Num Negative'] = np.sum(tst_sub_y == -1)
        tst_df.loc[i, 'Type'] = subpop_type
        tst_df.loc[i, 'Gini Impurity'] = 1. - (tst_frac_pos**2 + (1. - tst_frac_pos)**2)
        tst_df.loc[i, 'Avg Silhouette'] = np.mean(tst_silhouettes[tst_subpop_inds])
        tst_df.loc[i, 'Davies Bouldin'] = sklearn.metrics.davies_bouldin_score(X_test, tst_dummy_cluster_labels)
        tst_df.loc[i, 'Avg Distance Ratio'] = np.mean(pdist(tst_sub_x)) / np.mean(cdist(tst_sub_x, tst_nsub_x))
        tst_df.loc[i, 'Linear Sep Score'] = sklearn.svm.LinearSVC(C=10000.0).fit(X_test, tst_dummy_svm_labels).score(test_sub_x, tst_dummy_svm_labels[tst_sbcl]) # score assuming |subpop| << |all data|
        if subpop_type == 'feature':
            tst_df.loc[i, 'Semantic Info'] = str(descs[i]).replace("'", '"')

trn_df.to_csv('files/data/{}_trn_{}_desc.csv'.format(dataset_name, subpop_type), index=False)
if generate_tst_desc:
    tst_df.to_csv('files/data/{}_tst_{}_desc.csv'.format(dataset_name, subpop_type), index=False)
