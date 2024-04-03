import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)

import pandas as pd
import numpy as np
import argparse

COMBINED_CSV = 'combined.csv'

def main():
    parser = argparse.ArgumentParser(description='Combine lowerbound experiemnt results')
    parser.add_argument('directory', type=str, help='Directory to merge files for')
    args = parser.parse_args()

    # walk through the directory and find all the combined.csv files
    combined_data = pd.DataFrame()
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file == COMBINED_CSV and root != args.directory:
                combined_data = combined_data.append(pd.read_csv(os.path.join(root, file)), sort=False)

    print(combined_data.shape)
    combined_data.to_csv(os.path.join(args.directory, COMBINED_CSV), index=False)

if __name__ == '__main__':
    main()