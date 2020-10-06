import re
import sys
import json
import spacy
import string
import argparse

import numpy as np

from pingouin import mwu
from nltk.corpus import stopwords


def _process_strings(line,
                     lang_nlp,
                     get_lemmas,
                     get_pos,
                     remove_stopwords,
                     replace_stopwords,
                     get_maps):

    """ Helper function for obtaining various word representations """

    # strip, replace special tokens
    orig_line = line
    line = line.strip()
    line = re.sub(r'&apos;', '\'', line.strip())
    line = re.sub(r'&quot;', '\"', line.strip())
    # Tokenize etc.
    line_nlp = lang_nlp(line)
    spacy_tokens = [elem.text for elem in line_nlp]
    spacy_tokens_lower = [elem.text.lower() for elem in line_nlp]
    spacy_lemmas = None
    spacy_pos = None
    if get_lemmas:
        spacy_lemmas = list()
        for elem in line_nlp:
            if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
                spacy_lemmas.append(elem.lower_)
            else:
                spacy_lemmas.append(elem.lemma_.lower().strip())
    if get_pos:
        spacy_pos = [elem.pos_ for elem in line_nlp]

    # Generate a mapping between whitespace tokens and SpaCy tokens
    ws_tokens = orig_line.strip().split()
    ws_tokens_lower = orig_line.strip().lower().split()
    ws_to_spacy_map = dict()
    spacy_to_ws_map = dict()
    if get_maps:
        ws_loc = 0
        ws_tok = ws_tokens[ws_loc]

        for spacy_loc, spacy_tok in enumerate(spacy_tokens):
            while True:
                # Map whitespace tokens to be identical to spacy tokens
                ws_tok = re.sub(r'&apos;', '\'', ws_tok)
                ws_tok = re.sub(r'&quot;', '\"', ws_tok)

                if spacy_tok == ws_tok or spacy_tok in ws_tok:
                    # Terminate
                    if ws_loc >= len(ws_tokens):
                        break

                    # Extend maps
                    if not ws_to_spacy_map.get(ws_loc, None):
                        ws_to_spacy_map[ws_loc] = list()
                    ws_to_spacy_map[ws_loc].append(spacy_loc)
                    if not spacy_to_ws_map.get(spacy_loc, None):
                        spacy_to_ws_map[spacy_loc] = list()
                    spacy_to_ws_map[spacy_loc].append(ws_loc)

                    # Move pointer
                    if spacy_tok == ws_tok:
                        ws_loc += 1
                        if ws_loc < len(ws_tokens):
                            ws_tok = ws_tokens[ws_loc]
                    else:
                        ws_tok = ws_tok[len(spacy_tok):]
                    break
                else:
                    ws_loc += 1

        # Assert full coverage of whitespace and SpaCy token sequences by the mapping
        ws_covered = sorted(list(ws_to_spacy_map.keys()))
        spacy_covered = sorted(list(set(list([val for val_list in ws_to_spacy_map.values() for val in val_list]))))
        assert ws_covered == [n for n in range(len(ws_tokens))], \
            'WS-SpaCy mapping does not cover all whitespace tokens: {}; number of tokens: {}'\
            .format(ws_covered, len(ws_tokens))
        assert spacy_covered == [n for n in range(len(spacy_tokens))], \
            'WS-SpaCy mapping does not cover all SpaCy tokens: {}; number of tokens: {}' \
            .format(spacy_covered, len(spacy_tokens))

    if remove_stopwords:
        # Filter out stopwords
        nsw_spacy_tokens_lower = list()
        nsw_spacy_lemmas = list()
        for tok_id, tok in enumerate(spacy_tokens_lower):
            if tok not in STOP_WORDS:
                nsw_spacy_tokens_lower.append(tok)
                if get_lemmas:
                    nsw_spacy_lemmas.append(spacy_lemmas[tok_id])
            else:
                if replace_stopwords:
                    nsw_spacy_tokens_lower.append('<STPWRD>')
                    if get_lemmas:
                        nsw_spacy_lemmas.append('<STPWRD>')

        spacy_tokens_lower = nsw_spacy_tokens_lower
        if get_lemmas:
            spacy_lemmas = nsw_spacy_lemmas

    return line_nlp, spacy_tokens_lower, spacy_lemmas, spacy_pos, ws_tokens, ws_tokens_lower, ws_to_spacy_map, \
        spacy_to_ws_map


def _get_sentence_scores(seed_sentence, seed_parses, attractor_token_freq_dict, attractor_token_pmi_dict):
    """ Helper function for computing sentence-level scores """

    # Lemmatize seed sentence and compute sentence-level attractor scores
    spacy_tokens_lower = None
    if seed_parses is not None:
        if seed_parses.get(seed_sentence, None) is not None:
            spacy_tokens_lower, spacy_lemmas = seed_parses[seed_sentence]

    if spacy_tokens_lower is None:
        _, spacy_tokens_lower, spacy_lemmas, _, _, _, _, _ = \
            _process_strings(seed_sentence,
                             nlp,
                             get_lemmas=False,
                             get_pos=False,
                             remove_stopwords=False,
                             replace_stopwords=False,
                             get_maps=False)
        if seed_parses is not None:
            seed_parses[seed_sentence] = (spacy_tokens_lower, spacy_lemmas)

    # NOTE: PPMI is used in place of PMI to account for sentences containing no known attractors
    attractor_token_freq_scores = list()
    attractor_token_pmi_scores = list()
    attractor_token_ppmi_scores = list()

    for token in spacy_tokens_lower:
        token_freq = attractor_token_freq_dict.get(token, None)
        if token_freq is not None:
            attractor_token_freq_scores.append(token_freq)

        token_pmi = attractor_token_pmi_dict.get(token, None)
        if token_pmi is not None:
            attractor_token_pmi_scores.append(token_pmi)
            attractor_token_ppmi_scores.append(max(0, token_pmi))

    return [attractor_token_freq_scores, attractor_token_pmi_scores, attractor_token_ppmi_scores,
            len(spacy_tokens_lower), seed_parses]


def test_correlations(positive_samples_path, negative_samples_path, attractors_table_path):
    """ Computes possible correlations between seed sentence and attractor properties, scores them for significance,
    and produces corresponding visualizations """

    # Read-in sample tables
    print('Reading-in tables ...')
    with open(positive_samples_path, 'r', encoding='utf8') as psp:
        positive_samples = json.load(psp)
    with open(negative_samples_path, 'r', encoding='utf8') as nsp:
        negative_samples = json.load(nsp)

    # Read-in attractor table
    print('Reading-in attractor table ...')
    with open(attractors_table_path, 'r', encoding='utf8') as atp:
        attractors_table = json.load(atp)

    # Find lowest PMI value
    pmi_min_per_term = list()
    for term in attractors_table.keys():
        pmi_min_per_sense = list()
        for sense in attractors_table[term].keys():
            pmi_min_per_sense. \
                append(min([tpl[1] for tpl in attractors_table[term][sense]['[SORTED ATTRACTORS BY PMI]']]))
        pmi_min_per_term.append(min(pmi_min_per_sense))
    pmi_lower_bound = min(pmi_min_per_term)

    # Initialize score cache
    seed_parses = dict()
    seed_score_cache = dict()

    metrics = ['[SENT. SEED ATTRACTOR TOKEN FREQ]',
               '[SENT. SEED ATTRACTOR TOKEN PMI]',
               '[SENT. SEED ATTRACTOR TOKEN PPMI]',
               '[SENT. ADV ATTRACTOR TOKEN FREQ]',
               '[SENT. ADV ATTRACTOR TOKEN PMI]',
               '[SENT. ADV ATTRACTOR TOKEN PPMI]',
               '[SENT. ADV-SEED ATTRACTOR TOKEN FREQ DIFF]',
               '[SENT. ADV-SEED ATTRACTOR TOKEN PMI DIFF]',
               '[SENT. ADV-SEED ATTRACTOR TOKEN PPMI DIFF]',
               '[SEED SENTENCE LENGTH]']

    # Collect scores
    print('Looking up scores ...')
    positive_scores = {m: list() for m in metrics}
    negative_scores = {m: list() for m in metrics}

    # Restrict generation strategies
    # generation_strategies = []
    generation_strategies = ['insert_at_homograph', 'replace_at_homograph']

    # Declare function used to compute sentence-level scores
    # sent_fun = np.mean
    sent_fun = sum

    for samples, scores, path in [(positive_samples, positive_scores, positive_samples_path),
                                  (negative_samples, negative_scores, negative_samples_path)]:
        for term in samples.keys():
            print('Looking-up the term \'{:s}\''.format(term))
            for seed_cluster in samples[term].keys():

                # Compute sentence-level scores for the relevant cluster
                seed_sorted_attractor_freq = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY FREQ]']
                seed_sorted_attractor_pmi = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY PMI]']
                seed_attractor_token_freq_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_freq}
                seed_attractor_token_pmi_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_pmi}

                for adv_cluster in samples[term][seed_cluster].keys():

                    # Compute sentence-level scores for the relevant cluster
                    adv_sorted_attractor_freq = attractors_table[term][adv_cluster]['[SORTED ATTRACTORS BY FREQ]']
                    adv_sorted_attractor_pmi = attractors_table[term][adv_cluster]['[SORTED ATTRACTORS BY PMI]']
                    adv_attractor_token_freq_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in adv_sorted_attractor_freq}
                    adv_attractor_token_pmi_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in adv_sorted_attractor_pmi}

                    for seed_sentence in samples[term][seed_cluster][adv_cluster].keys():
                        for sample in samples[term][seed_cluster][adv_cluster][seed_sentence]:
                            seed_sentence = sample[2].strip()
                            adv_sentence = sample[0].strip()

                            # Skip samples obtained through disregarded generation strategies
                            if 'attractors' not in path:
                                if len(generation_strategies) > 0:
                                    if sample[-2][-1] not in generation_strategies:
                                        continue

                            is_insert = 'insert' in sample[-2][-1]

                            # Look up attractor scores
                            if is_insert:
                                sample_scores = sample[7]
                                seed_attr_freq = seed_attractor_token_freq_dict.get(sample[6], 0)
                                adv_attr_freq = sample_scores['[SORTED ATTRACTORS BY FREQ]']

                                seed_attr_pmi = seed_attractor_token_pmi_dict.get(sample[6], pmi_lower_bound)
                                adv_attr_pmi = sample_scores['[SORTED ATTRACTORS BY PMI]']

                                seed_attr_ppmi = max(0., seed_attractor_token_pmi_dict.get(sample[6], 0.))
                                adv_attr_ppmi = max(0., sample_scores['[SORTED ATTRACTORS BY PMI]'])
                            else:
                                seed_attr_freq = 0
                                adv_attr_freq = 0
                                seed_attr_pmi = 0
                                adv_attr_pmi = 0
                                seed_attr_ppmi = 0
                                adv_attr_ppmi = 0

                            # Compute sentence-level scores
                            if is_insert:
                                seed_scores = seed_score_cache.get((seed_sentence, term, seed_cluster), None)
                                if seed_scores is None:
                                    seed_scores = _get_sentence_scores(seed_sentence, seed_parses,
                                                                       seed_attractor_token_freq_dict,
                                                                       seed_attractor_token_pmi_dict)
                                    seed_score_cache[(seed_sentence, term, seed_cluster)] = seed_scores[:-1]
                                    seed_parses = seed_scores[-1]
                            else:
                                seed_scores = _get_sentence_scores(adv_sentence, None,
                                                                   seed_attractor_token_freq_dict,
                                                                   seed_attractor_token_pmi_dict)

                            if is_insert:
                                adv_scores = seed_score_cache.get((seed_sentence, term, adv_cluster), None)
                                if adv_scores is None:
                                    adv_scores = _get_sentence_scores(seed_sentence, seed_parses,
                                                                      adv_attractor_token_freq_dict,
                                                                      adv_attractor_token_pmi_dict)
                                    seed_score_cache[(seed_sentence, term, adv_cluster)] = adv_scores[:-1]
                                    seed_parses = adv_scores[-1]
                            else:
                                adv_scores = _get_sentence_scores(adv_sentence, None,
                                                                  adv_attractor_token_freq_dict,
                                                                  adv_attractor_token_pmi_dict)

                            # Extend score tables
                            is_insert = int(is_insert)
                            if len(seed_scores[0]) > 0:
                                seed_freq = sent_fun(seed_scores[0])
                                seed_ppmi = sent_fun(seed_scores[2])
                                scores['[SENT. SEED ATTRACTOR TOKEN PMI]'] \
                                    .append((sent_fun(seed_scores[1]) + seed_attr_pmi) / (seed_scores[3] + is_insert))
                            else:
                                seed_freq = 0
                                seed_ppmi = 0
                                scores['[SENT. SEED ATTRACTOR TOKEN PMI]'].append(pmi_lower_bound)

                            scores['[SENT. SEED ATTRACTOR TOKEN FREQ]']\
                                .append((seed_freq + seed_attr_freq) / (seed_scores[3] + is_insert))
                            scores['[SENT. SEED ATTRACTOR TOKEN PPMI]']\
                                .append((seed_ppmi + seed_attr_ppmi) / (seed_scores[3] + is_insert))

                            if len(adv_scores[0]) > 0:
                                adv_freq = sent_fun(adv_scores[0])
                                adv_ppmi = sent_fun(adv_scores[2])
                                scores['[SENT. ADV ATTRACTOR TOKEN PMI]'] \
                                    .append((sent_fun(adv_scores[1]) + adv_attr_pmi) / (seed_scores[3] + is_insert))
                            else:
                                adv_freq = 0
                                adv_ppmi = 0
                                scores['[SENT. ADV ATTRACTOR TOKEN PMI]'].append(pmi_lower_bound)

                            scores['[SENT. ADV ATTRACTOR TOKEN FREQ]']\
                                .append((adv_freq + adv_attr_freq) / (seed_scores[3] + is_insert))
                            scores['[SENT. ADV ATTRACTOR TOKEN PPMI]']\
                                .append((adv_ppmi + adv_attr_ppmi) / (seed_scores[3] + is_insert))

                            scores['[SENT. ADV-SEED ATTRACTOR TOKEN FREQ DIFF]']\
                                .append(scores['[SENT. ADV ATTRACTOR TOKEN FREQ]'][-1] -
                                        scores['[SENT. SEED ATTRACTOR TOKEN FREQ]'][-1])
                            scores['[SENT. ADV-SEED ATTRACTOR TOKEN PMI DIFF]'] \
                                .append(scores['[SENT. ADV ATTRACTOR TOKEN PMI]'][-1] -
                                        scores['[SENT. SEED ATTRACTOR TOKEN PMI]'][-1])
                            scores['[SENT. ADV-SEED ATTRACTOR TOKEN PPMI DIFF]']\
                                .append(scores['[SENT. ADV ATTRACTOR TOKEN PPMI]'][-1] -
                                        scores['[SENT. SEED ATTRACTOR TOKEN PPMI]'][-1])

                            scores['[SEED SENTENCE LENGTH]'].append(seed_scores[3])

    # Calculate correlation values
    correlation_values = dict()
    print('Computing correlations ...')
    for metric_key in metrics:
        print('Metric: {:s}'.format(metric_key))
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
    num_pos = len(positive_scores['[SEED SENTENCE LENGTH]'])
    num_neg = len(negative_scores['[SEED SENTENCE LENGTH]'])
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

    # Instantiate processing pipeline
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat', 'parser'])
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    STOP_WORDS = stopwords.words('english')

    test_correlations(args.positive_samples_path, args.negative_samples_path, args.attractors_table_path)

