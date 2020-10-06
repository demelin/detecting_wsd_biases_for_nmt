import sys
import json
import string
import argparse

import numpy as np
from pingouin import mwu


def test_correlations(positive_samples_path, negative_samples_path, attractors_table_path):
    """ Computes possible correlations between attractor importance (and other) metrics, scores them for significance,
    and produces corresponding visualizations """

    # Read-in tables
    print('Reading-in tables ...')
    with open(positive_samples_path, 'r', encoding='utf8') as psp:
        positive_samples = json.load(psp)
    with open(negative_samples_path, 'r', encoding='utf8') as nsp:
        negative_samples = json.load(nsp)

    # Read-in attractor table
    print('Reading-in attractor table ...')
    with open(attractors_table_path, 'r', encoding='utf8') as atp:
        attractors_table = json.load(atp)

    # Declare attractor importance metrics to consider
    metrics = ['[HOMOGRAPH TOTAL FREQUENCY]',
               '[HOMOGRAPH SEED FREQUENCY]',
               '[HOMOGRAPH ADV FREQUENCY]',
               '[HOMOGRAPH FREQ DIFF]']

    # Collect scores
    print('Looking up scores ...')
    positive_scores = {m: list() for m in metrics}
    negative_scores = {m: list() for m in metrics}
    for term in positive_samples.keys():
        total_attractor_freq = sum([len(attractors_table[term][sense_cluster]['[SENTENCE PAIRS]'])
                                    for sense_cluster in attractors_table[term].keys()])
        for seed_cluster in positive_samples[term].keys():
            seed_attractor_freq = len(attractors_table[term][seed_cluster]['[SENTENCE PAIRS]'])
            for adv_cluster in positive_samples[term][seed_cluster].keys():
                adv_attractor_freq = len(attractors_table[term][adv_cluster]['[SENTENCE PAIRS]'])
                for seed_sentence in positive_samples[term][seed_cluster][adv_cluster].keys():
                    for sample in positive_samples[term][seed_cluster][adv_cluster][seed_sentence]:

                        # Only consider samples derived from correctly translated seeds
                        if 'true_samples' not in positive_samples_path:
                            if 'attractors' not in positive_samples_path:
                                if sample[-1][0] != 'not_flipped':
                                    continue

                        positive_scores['[HOMOGRAPH TOTAL FREQUENCY]'].append(total_attractor_freq)
                        positive_scores['[HOMOGRAPH SEED FREQUENCY]'].append(seed_attractor_freq)
                        if 'bad_translations' not in positive_samples_path:
                            positive_scores['[HOMOGRAPH ADV FREQUENCY]'].append(adv_attractor_freq)
                        else:
                            translation_clusters = list(set(sample[2]))
                            tc_frequencies = list()
                            for tc in translation_clusters:
                                # Needed to skip clusters for which no attractors are known
                                if attractors_table[term].get(tc, None) is None:
                                    continue
                                tc_frequencies.append(len(attractors_table[term][tc]['[SENTENCE PAIRS]']))
                            adv_attractor_freq = max(tc_frequencies)
                            positive_scores['[HOMOGRAPH ADV FREQUENCY]'].append(adv_attractor_freq)
                        positive_scores['[HOMOGRAPH FREQ DIFF]'].append(adv_attractor_freq - seed_attractor_freq)

    for term in negative_samples.keys():
        total_attractor_freq = sum([len(attractors_table[term][sense_cluster]['[SENTENCE PAIRS]'])
                                    for sense_cluster in attractors_table[term].keys()])
        for seed_cluster in negative_samples[term].keys():
            seed_attractor_freq = len(attractors_table[term][seed_cluster]['[SENTENCE PAIRS]'])
            for adv_cluster in negative_samples[term][seed_cluster].keys():
                adv_attractor_freq = len(attractors_table[term][adv_cluster]['[SENTENCE PAIRS]'])
                for seed_sentence in negative_samples[term][seed_cluster][adv_cluster].keys():
                    for sample in negative_samples[term][seed_cluster][adv_cluster][seed_sentence]:

                        # Only consider samples derived from correctly translated seeds
                        if 'true_samples' not in negative_samples_path:
                            if 'attractors' not in negative_samples_path:
                                if sample[-1][0] != 'not_flipped':
                                    continue

                        negative_scores['[HOMOGRAPH TOTAL FREQUENCY]'].append(total_attractor_freq)
                        negative_scores['[HOMOGRAPH SEED FREQUENCY]'].append(seed_attractor_freq)
                        if 'bad_translations' not in negative_samples_path:
                            negative_scores['[HOMOGRAPH ADV FREQUENCY]'].append(adv_attractor_freq)
                        else:
                            translation_clusters = list(set(sample[2]))
                            tc_frequencies = list()
                            for tc in translation_clusters:
                                # Needed to skip clusters for which no attractors are known
                                if attractors_table[term].get(tc, None) is None:
                                    continue
                                tc_frequencies.append(len(attractors_table[term][tc]['[SENTENCE PAIRS]']))
                            adv_attractor_freq = max(tc_frequencies)
                            negative_scores['[HOMOGRAPH ADV FREQUENCY]'].append(adv_attractor_freq)
                        negative_scores['[HOMOGRAPH FREQ DIFF]'].append(adv_attractor_freq - seed_attractor_freq)

    # Calculate correlation values
    correlation_values = dict()
    print('Computing correlations ...')
    for metric_key in metrics:
        correlation_values[metric_key] = dict()
        positive_metric_scores = positive_scores[metric_key]
        negative_metric_scores = negative_scores[metric_key]
        # Perform the Mann–Whitney U test
        mwu_df = mwu(negative_metric_scores, positive_metric_scores, tail='two-sided')
        mwu_df_rev = mwu(positive_metric_scores, negative_metric_scores, tail='two-sided')
        correlation_values[metric_key]['MWU'] = mwu_df
        correlation_values[metric_key]['MWU_rev'] = mwu_df_rev
        # Add mean (addition indication of the effect size)
        correlation_values[metric_key]['MEANS'] = (np.mean(positive_metric_scores), np.mean(negative_metric_scores),
                                                   np.mean(positive_metric_scores) - np.mean(negative_metric_scores))

    # Report results

    # Compute threshold for effect size interpretation
    num_pos = len(positive_scores['[HOMOGRAPH TOTAL FREQUENCY]'])
    num_neg = len(negative_scores['[HOMOGRAPH TOTAL FREQUENCY]'])
    base_pos = num_pos / (num_pos + num_neg)
    base_neg = num_neg / (num_pos + num_neg)

    small_threshold = 0.2 / np.sqrt(0.2 ** 2 + (1 / (base_pos * base_neg)))
    moderate_threshold = 0.5 / np.sqrt(0.5 ** 2 + (1 / (base_pos * base_neg)))
    max_threshold = 0.8 / np.sqrt(0.8 ** 2 + (1 / (base_pos * base_neg)))

    print('-' * 20)
    print('RESULTS: ')
    for metric_key in metrics:
        print(metric_key)
        for measure in ['MWU', 'MEANS']:
            if measure == 'MEANS':
                values = list()
                for v in correlation_values[metric_key][measure]:
                    values.append(float('{:.4f}'.format(v)))
                print(measure, ' ', values)
            else:
                u = correlation_values[metric_key][measure].iloc[0]['U-val']
                u_rev = correlation_values[metric_key]['MWU_rev'].iloc[0]['U-val']
                p = correlation_values[metric_key][measure].iloc[0]['p-val']
                p = p if p > 0.00005 else 0.0
                rbc = correlation_values[metric_key][measure].iloc[0]['RBC']
                # cles = correlation_values[metric_key][measure].iloc[0]['CLES']
                aw = ((num_pos * num_neg) - u) / (num_pos * num_neg)
                aw_rev = ((num_pos * num_neg) - u_rev) / (num_pos * num_neg)
                print('MWU (u / p) : {:.3f}, {:.4f}'.format(u, p))
                print('MWU (rbc) : {:.3f}'.format(rbc))
                print('MWU (Aw) : {:.3f} | {:.3f}'.format(aw, aw_rev))
        print('-' * 10)
    print('Thresholds: {:.4f} | {:.4f} | {:.4f}'.format(small_threshold, moderate_threshold, max_threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_samples_path', type=str, help='path to the file containing successful adversarial samples',
                        required=True)
    parser.add_argument('--negative_samples_path', type=str, help='path to the file containing unsuccessful adversarial samples',
                        required=True)
    parser.add_argument('--attractors_table_path', type=str, help='path to the attractor pool',
                        required=True)
    args = parser.parse_args()

    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    test_correlations(args.positive_samples_path, args.negative_samples_path, args.attractors_table_path)

