import sys
import json
import argparse

import pandas as pd
from pingouin import chi2_independence


def test_significance(positive_samples_path, negative_samples_path, good_seed_only):
    """ Computes the significance levels / effect size of the generation strategy on the success of adversarial
    samples """

    # Read-in tables
    print('Reading-in tables ...')
    with open(positive_samples_path, 'r', encoding='utf8') as psp:
        positive_samples = json.load(psp)
    with open(negative_samples_path, 'r', encoding='utf8') as nsp:
        negative_samples = json.load(nsp)

    # Store success labels per generation strategy
    success_var = {'insert_at_homograph': list(),
                   'replace_at_homograph': list(),
                   'insert_at_other': list(),
                   'replace_at_other': list()}

    # Construct dataframe for the Chi^2 test
    print('Looking up sample provenance ...')
    for term in positive_samples.keys():
        for seed_cluster in positive_samples[term].keys():
            for adv_cluster in positive_samples[term][seed_cluster].keys():
                for seed_sentence in positive_samples[term][seed_cluster][adv_cluster].keys():
                    for sample in positive_samples[term][seed_cluster][adv_cluster][seed_sentence]:
                        if good_seed_only == 'True' and sample[20][0] != 'not_flipped':
                            continue
                        gen_strat = sample[19][-1]
                        success_var[gen_strat].append(1)
    for term in negative_samples.keys():
        for seed_cluster in negative_samples[term].keys():
            for adv_cluster in negative_samples[term][seed_cluster].keys():
                for seed_sentence in negative_samples[term][seed_cluster][adv_cluster].keys():
                    for sample in negative_samples[term][seed_cluster][adv_cluster][seed_sentence]:
                        if good_seed_only == 'True' and sample[20][0] != 'not_flipped':
                            continue
                        gen_strat = sample[19][-1]
                        success_var[gen_strat].append(0)

    # Construct dataframe
    print('Computing correlations ...')
    success_dict = {'method': list(), 'labels': list()}
    for m in success_var.keys():
        success_dict['method'] += [m] * len(success_var[m])
        success_dict['labels'] += success_var[m]
    unrolled_success_dict = pd.DataFrame.from_dict(success_dict)
    # Perform Chi^2 test
    expected, observed, stats = chi2_independence(unrolled_success_dict, x='method', y='labels')
    chi2 = stats.iloc[0]['chi2']
    p = stats.iloc[0]['p']
    p = p if p > 0.00005 else 0.0
    v = stats.iloc[0]['cramer']

    # Report
    print('Done!')
    print('=' * 20)
    print('CHI^2 STATS:')
    if p > 0.0:
        print('{:.3f}, {:.4f}, {:.4f}'.format(chi2, p, v))
    else:
        print('{:.3f}, {:.1f}, {:.4f}'.format(chi2, p, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_samples_path', type=str, help='path to the file containing successful adversarial samples',
                        required=True)
    parser.add_argument('--negative_samples_path', type=str, help='path to the file containing unsuccessful adversarial samples',
                        required=True)
    parser.add_argument('--good_seed_only', type=str, help='whether to only consider samples with correct seed translations (set to True, if so)',
                        required=True)
    args = parser.parse_args()
    test_significance(args.positive_samples_path, args.negative_samples_path, args.good_seed_only)

