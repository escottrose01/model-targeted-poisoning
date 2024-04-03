import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)

import pandas as pd
import numpy as np
import argparse

ATK_NPZ_FORMAT = 'subpop-{0}-atk.npz'
SUMMARY_CSV = 'subpop_desc.csv'
LOWERBOUNDS_CSV = 'lowerbounds.csv'
COMBINED_CSV = 'combined.csv'

def main():
    parser = argparse.ArgumentParser(description='Combine lowerbound experiemnt results')
    parser.add_argument('directory', type=str, help='Directory to merge files for')
    args = parser.parse_args()

    # Load lowerbounds.csv and subpop_desc.csv
    lowerbounds_file = os.path.join(args.directory, LOWERBOUNDS_CSV)
    subpop_desc_file = os.path.join(args.directory, SUMMARY_CSV)
    combined_data = combine_csv_files(lowerbounds_file, subpop_desc_file)

    # Load the attack results
    for subpop_ix, row in combined_data.iterrows():
        try:
            attack_file = os.path.join(args.directory, ATK_NPZ_FORMAT.format(subpop_ix))
            attack_results = np.load(attack_file)

            attack_stats = attack_results['attack_stats'].item()
            for k, v in attack_stats.items():
                combined_data.loc[subpop_ix, k] = v

            attack_log = attack_results['attack_log']
            inf_lossdiff = float('inf')
            mtp_lossdiff = float('inf')
            for atk in attack_log:
                if atk['attack_tag'] == 'influence':
                    inf_lossdiff = min(inf_lossdiff, atk['loss_diff'])
                else:
                    mtp_lossdiff = min(mtp_lossdiff, atk['loss_diff'])
            combined_data.loc[subpop_ix, 'influence_loss_diff'] = inf_lossdiff
            combined_data.loc[subpop_ix, 'mtp_loss_diff'] = mtp_lossdiff

        except IOError:
            print('File not found for subpop {}'.format(subpop_ix))
            continue

    # Save the combined data
    combined_data.to_csv(os.path.join(args.directory, COMBINED_CSV), index=False)

def combine_csv_files(lowerbounds_file, subpop_desc_file):
    # Load lowerbounds.csv and subpop_desc.csv
    lowerbounds = pd.read_csv(lowerbounds_file)
    subpop_desc = pd.read_csv(subpop_desc_file)
    assert len(lowerbounds) == len(subpop_desc), 'Number of rows in lowerbounds.csv and subpop_desc.csv do not match'

    combined_data = pd.concat([subpop_desc, lowerbounds], axis=1)
    return combined_data

if __name__ == '__main__':
    main()
