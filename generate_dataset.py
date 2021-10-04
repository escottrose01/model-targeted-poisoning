import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dset_type', default='make_classification', help='sklearn dataset type')
parser.add_argument('--rand_seed', default=0, type=int, help='random seed for dataset randomness')
parser.add_argument('--n_samples', default=3000, type=int, help='total number of samples to generate')

# make_classification params
parser.add_argument('--flip_y', default=0.0, type=float, help='flip_y param for make_classification dataset')
parser.add_argument('--class_sep', default=1.0, type=float, help='class_sep param for make_classification dataset')
parser.add_argument('--n_clusters_per_class', default=1, type=int, help='n_clusters_per_class param for make_classification dataset')

args = parser.parse_args()

dset_type = args.dset_type
if (dset_type == 'make_classification'):
    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=args.n_clusters_per_class,
        weights=[0.5],
        random_state=args.rand_seed,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
    )

    params = {
        'flip_y':args.flip_y,
        'class_sep':args.class_sep,
        'n_clusters_per_class':args.n_clusters_per_class
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3., random_state=1)

    # labels to [-1, 1]
    y_test = 2*y_test - 1
    y_train = 2*y_train - 1

    # features to [0, 1]
    lo = np.min(X_train, axis=0)
    hi = np.max(X_train, axis=0)
    ptp = hi - lo
    ptp[ptp == 0] = 1. # prevent division by zero

    X_train = (X_train - lo) / ptp
    X_test = (X_test - lo) / ptp
else:
    sys.exit('Unsupported dataset type!')

out_npz = 'files/data/synthetic_train_test.npz'
out_json = 'files/data/synthetic.json'
np.savez(
    file=out_npz,
    X_train=X_train,
    Y_train=y_train,
    X_test=X_test,
    Y_test=y_test
)
with open(out_json, 'w') as outfile:
    json.dump(params, outfile)

print('Successfuly generated a {} dataset with {} pts'.format(args.dset_type, args.n_samples))
