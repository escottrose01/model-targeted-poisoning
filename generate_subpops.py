from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import argparse
import os
import csv
from sklearn import cluster

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='adult', help='four datasets: mnist_17, adult, 2d_toy, dogfish')
parser.add_argument('--subpop_type', default='cluster', choices=['cluster', 'feature'], help='subpopulaton type: cluster or feature')
parser.add_argument('--subpop_ratio', default='0.05', help='desired subpopulation ratio for feature subpop: between 0 and 1')
parser.add_argument('--num_clusters', default='auto', help='number of clusters to use for clustering subpop: integer or \'auto\' for default')

args = parser.parse_args()

NUM_CLUSTERS = {
    'dogfish' : 3,
    'mnist_17' : 20,
    'adult' : 20,
    '2d_toy' : 4
}

assert args.dataset in ['dogfish', 'mnist_17', 'adult', '2d_toy'], 'not a valid dataset!'
dataset_name = args.dataset

subpop_type = args.subpop_type

assert (args.num_clusters == 'auto') or (args.num_clusters.isdigit()), 'not a valid num_clusters argument!'
num_clusters = NUM_CLUSTERS[dataset_name] if args.num_clusters == 'auto' else int(args.num_clusters)

try:
    subpop_ratio = float(args.subpop_ratio)
except ValueError:
    assert False, 'not a valid subpop_ratio argument!'



subpop_fname = 'files/data/{}_{}_labels.txt'.format(dataset_name, subpop_type)
if os.path.isfile(subpop_fname):
    sys.exit(0)

X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)

if min(Y_test) > -1:
    Y_test = 2*Y_test - 1
if min(Y_train) > -1:
    Y_train = 2*Y_train - 1

if subpop_type == 'cluster':
    km = cluster.KMeans(n_clusters=num_clusters, random_state=0)
    km.fit(X_train)
    trn_km = km.labels_
    tst_km = km.predict(X_test)

    trn_all_subpops = [[x] for x in list(trn_km)]
    tst_all_subpops = [[x] for x in list(tst_km)]
elif subpop_type == 'feature':
    assert dataset_name == 'adult', 'feature matching is only supported for adult dataset!'
    _, _ , _, _, all_cols = load_dataset(dataset_name + '_unscaled')

    feature_inds = [
        list(range(4, 12)),     # work class
        list(range(12, 27)),    # education level
        list(range(27, 33)),    # marital status
        list(range(33, 47)),    # occupation
        list(range(47, 52)),    # relationship status
        list(range(52, 56)),    # race
        list(range(56, 57)),    # sex
    ]

    feature_descriptions = [
        'work class',
        'education',
        'marital status',
        'occupation',
        'relationship status',
        'race',
        'sex'
    ]

    trn_all_subpops = [[] for i in range(X_train.shape[0])] # -1 for nonempty lists
    tst_all_subpops = [[] for i in range(X_test.shape[0])]
    subcl_ix = 0

    # we generate subpopulations by feature matching subsets of features
    feature_subsets = []
    for i in range(1, 1 << len(feature_inds)):
        feature_subsets.append([j for j in range(len(feature_inds)) if (1 << j)&i])

    for feature_subset in feature_subsets:
        subcl_inds_nested = [feature_inds[i] for i in feature_subset]
        if (len(subcl_inds_nested) > 3): continue
        subcl_inds = [item for sublist in subcl_inds_nested for item in sublist]
        trn_prot = X_train[:, subcl_inds]
        tst_prot = X_test[:, subcl_inds]
        prot_cols = [all_cols[i] for i in range(len(all_cols)) if i in subcl_inds]
        subclasses, counts = np.unique(trn_prot, axis=0, return_counts=True)

        count_min = (subpop_ratio - 0.001) * X_train.shape[0]
        count_max = (subpop_ratio + 0.001) * X_train.shape[0]

        for subcl, count in zip(subclasses, counts):
            if (count_min < count < count_max):
                trn_subcl = np.where(np.linalg.norm(trn_prot - subcl, axis=1) == 0)[0]
                tst_subcl = np.where(np.linalg.norm(tst_prot - subcl, axis=1) == 0)[0]

                for ix in trn_subcl:
                    trn_all_subpops[ix].append(subcl_ix)
                for ix in tst_subcl:
                    tst_all_subpops[ix].append(subcl_ix)
                subcl_ix += 1

                print('comparing', [feature_descriptions[i] for i in range(len(feature_descriptions)) if i in feature_subset], ', found match filtering on')
                print('\t', [prot_col for v, prot_col in zip(subcl, prot_cols) if v > 0.5], count)
                print('\t% of train data:', trn_subcl.shape[0]/X_train.shape[0])
                print('\t% of test data: ', tst_subcl.shape[0]/X_test.shape[0])

# save the subpop info to ensure everything is reproducible
delim = str(' ').encode('utf-8') # convert to utf-8
subpop_fname = 'files/data/{}_trn_{}_labels.txt'.format(dataset_name, subpop_type)
with open(subpop_fname, 'w') as f:
    csv.writer(f, delimiter=delim).writerows(trn_all_subpops)
subpop_fname = 'files/data/{}_tst_{}_labels.txt'.format(dataset_name, subpop_type)
with open(subpop_fname, 'w') as f:
    csv.writer(f, delimiter=delim).writerows(tst_all_subpops)
