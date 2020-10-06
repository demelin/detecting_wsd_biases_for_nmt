import re
import sys
import json
import spacy
import string
import random
import argparse

import numpy as np
from nltk.corpus import stopwords


def build_corpus(adversarial_samples_path, attractors_table_path, corpus_size,
                 max_homograph_count, constrain_sampling_region):
    """ Creates a WSD challenge corpus from the totality of the identified seed sentences. """

    # Read-in adversarial samples
    print('Reading-in adversarial samples table ...')
    with open(adversarial_samples_path, 'r', encoding='utf8') as asp:
        adversarial_samples_table = json.load(asp)

    # Read-in attractor table
    print('Reading-in attractor table ...')
    with open(attractors_table_path, 'r', encoding='utf8') as atp:
        attractors_table = json.load(atp)

    # Initialize down-sampled sample table
    corpus_samples_table = dict()

    # Initialize seed score cache
    seen_seeds = dict()
    score_cache = dict()
    seed_to_sample = dict()

    # Pre-score seed sentences
    print('Looking up attractor scores ...')
    for term_id, term in enumerate(adversarial_samples_table.keys()):
        print('Looking-up the term \'{:s}\''.format(term))
        for seed_cluster in adversarial_samples_table[term].keys():
            seed_sense_freq = len(attractors_table[term][seed_cluster]['[SENTENCE PAIRS]'])
            for adv_cluster in adversarial_samples_table[term][seed_cluster].keys():
                for sample in adversarial_samples_table[term][seed_cluster][adv_cluster]:

                    adv_sample = sample[0].strip()
                    seed_sentence = sample[1].strip()

                    # Apply filtering that is consistent with the evaluation process
                    adv_src_tokens = [tok.strip(punctuation_plus_space) for tok in adv_sample.split()]
                    adv_src_tokens = [tok for tok in adv_src_tokens if len(tok) > 0]
                    if adv_src_tokens.count(sample[3].strip(punctuation_plus_space)) > 1:
                        continue
                    # Ignore samples with sentence-initial attractors
                    if 0 in sample[11]:
                        continue

                    true_src_tokens = [tok.strip(punctuation_plus_space) for tok in seed_sentence.split()]
                    true_src_tokens = [tok for tok in true_src_tokens if len(tok) > 0]
                    # Ignore short sentences
                    if len(true_src_tokens) < 10:
                        continue

                    if seen_seeds.get((seed_sentence, term, seed_cluster), None) is not None:
                        continue
                    seen_seeds[(seed_sentence, term, seed_cluster)] = True

                    adv_sense_freq = 0
                    for ac in attractors_table[term].keys():
                        if ac == seed_cluster:
                            continue
                        ac_freq = len(attractors_table[term][ac]['[SENTENCE PAIRS]'])
                        if ac_freq > adv_sense_freq:
                            adv_sense_freq = ac_freq

                    score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = \
                        adv_sense_freq - seed_sense_freq

                    seed_to_sample[(seed_sentence, term, seed_cluster, adv_cluster)] = sample

    # Sort term samples by attractor property
    score_dist = list(score_cache.values())
    # min: (high negative values) much stronger seed bias than adversarial bias
    # max: (low negative to positive values?) stronger adversarial bias than seed bias
    # sentences with PPMI of 0 are regarded as 'unbiased', which is a strong simplification
    if constrain_sampling_region == 'True':
        sample_bound = np.quantile(score_dist, 0.9)
    else:
        sample_bound = max(score_dist)

    # Sort term samples by attractor property
    print('Ranking and sampling adversarial samples ...')
    scored_seeds = [(k[0], k[1], k[2], k[3], v) for k, v in score_cache.items()]
    scored_seeds = sorted(scored_seeds, reverse=True, key=lambda x: len(x[0].split()))  # sort by length first
    sorted_seeds = sorted(scored_seeds, reverse=True, key=lambda x: x[-1])

    # Sample samples
    samples_drawn = 0
    diff_terms = 0
    diff_seeds = 0
    diff_seed_clusters = 0
    counts_per_homograph = dict()
    included_seeds = dict()

    # Assemble challenge corpus
    for ss in sorted_seeds:

        new_term = False
        new_seed = False
        new_seed_cluster = False

        # Don't exceed specified challenge set size
        if samples_drawn == corpus_size:
            break

        # Enforce diversity
        if max_homograph_count > 0 and ss[1] in counts_per_homograph.keys():
            if counts_per_homograph[ss[1]] >= max_homograph_count:
                continue

        # Avoid sampling from the extremes
        if ss[-1] > sample_bound:
            continue

        if included_seeds.get(ss[0], None) is None:
            included_seeds[ss[0]] = True
            new_seed = True

        # Grow challenge set
        if corpus_samples_table.get(ss[1], None) is None:
            corpus_samples_table[ss[1]] = dict()
            counts_per_homograph[ss[1]] = 0
            new_term = True
        if corpus_samples_table[ss[1]].get(ss[2], None) is None:
            corpus_samples_table[ss[1]][ss[2]] = dict()
            new_seed_cluster = True
        if corpus_samples_table[ss[1]][ss[2]].get(ss[3], None) is None:
            corpus_samples_table[ss[1]][ss[2]][ss[3]] = list()

        corpus_samples_table[ss[1]][ss[2]][ss[3]].append(seed_to_sample[(ss[0], ss[1], ss[2], ss[3])])
        counts_per_homograph[ss[1]] += 1
        samples_drawn += 1
        if new_term:
            diff_terms += 1
        if new_seed:
            diff_seeds += 1
        if new_seed_cluster:
            diff_seed_clusters += 1

    print('FINAL STATS')
    min_samples = float('inf')
    max_samples = 0
    print('=' * 20)
    print('Evaluated {:d} terms, extracted {:d} samples'
          .format(len(adversarial_samples_table.keys()), samples_drawn))
    print('Total unique terms: {:d}'.format(diff_terms))
    print('Mean unique seeds per term: {:.4f}'.format(diff_seeds / diff_terms))
    print('Mean unique seed clusters per term: {:.4f}'.format(diff_seed_clusters / diff_terms))
    print('=' * 20)
    print('Samples per term:')
    for term in corpus_samples_table.keys():
        term_entry_size = 0
        term_seeds = list()
        for seed_cluster in corpus_samples_table[term].keys():
            for adv_cluster in corpus_samples_table[term][seed_cluster].keys():
                term_entry_size += len(corpus_samples_table[term][seed_cluster][adv_cluster])
                for sample in corpus_samples_table[term][seed_cluster][adv_cluster]:
                    if sample[1] not in term_seeds:
                        term_seeds.append(sample[1])
        print('{:s}: {:d} samples, {:d} unique seeds'
              .format(term, term_entry_size, len(term_seeds)))
        min_samples = len(term_seeds) if len(term_seeds) < min_samples else min_samples
        max_samples = len(term_seeds) if len(term_seeds) > max_samples else max_samples
    print('=' * 20)
    print('Min samples per term: {:d}'.format(min_samples))
    print('Max samples per term: {:d}'.format(max_samples))

    # Save to JSON
    out_path = adversarial_samples_path[:-5] + \
        '_natural_challenge_from_homograph_freq_diff_{:d}_{:d}_{:s}.json'\
        .format(corpus_size, max_homograph_count, constrain_sampling_region)
    with open(out_path, 'w', encoding='utf8') as op:
        json.dump(corpus_samples_table, op, indent=3, sort_keys=True, ensure_ascii=False)
    print('Done!')
    print('Saved the natural challenge corpus table to {:s}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_samples_path', type=str, help='path to the adversarial sample pool',
                        required=True)
    parser.add_argument('--attractors_table_path', type=str, help='path to the attractor pool',
                        required=True)
    parser.add_argument('--corpus_size', type=int, help='size of the challenge set in pairs',
                        required=True)
    parser.add_argument('--max_homograph_count', type=int, help='max samples per homograph',
                        required=True)
    parser.add_argument('--constrain_sampling_region', type=str,
                        help='whether to filter sentences based on their homograph sense frequency')
    args = parser.parse_args()

    # Instantiate processing pipeline
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    STOP_WORDS = stopwords.words('english')

    build_corpus(args.adversarial_samples_path, args.attractors_table_path, args.corpus_size,
                 args.max_homograph_count, args.constrain_sampling_region)
