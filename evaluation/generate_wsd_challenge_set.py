import re
import sys
import json
import spacy
import string
import random
import argparse

import numpy as np
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


def _get_sentence_scores(seed_sentence, seed_parses,
                         attractor_token_freq_dict, attractor_token_pmi_dict,
                         attractor_lemma_freq_dict, attractor_lemma_pmi_dict):
    """ Helper function for computing sentence-level scores """

    # Lemmatize seed sentence and compute sentence-level attractor scores
    if seed_parses.get(seed_sentence, None) is not None:
        spacy_tokens_lower, spacy_lemmas = seed_parses[seed_sentence]
    else:
        _, spacy_tokens_lower, spacy_lemmas, _, _, _, _, _ = \
            _process_strings(seed_sentence,
                             nlp,
                             get_lemmas=True,
                             get_pos=False,
                             remove_stopwords=False,
                             replace_stopwords=False,
                             get_maps=False)
        seed_parses[seed_sentence] = (spacy_tokens_lower, spacy_lemmas)

    # NOTE: PPMI is used in place of PMI to account for sentences containing no known attractors
    attractor_token_freq_scores = list()
    attractor_token_ppmi_scores = list()
    for token in spacy_tokens_lower:
        token_freq = attractor_token_freq_dict.get(token, None)
        if token_freq is not None:
            attractor_token_freq_scores.append(token_freq)
        token_pmi = attractor_token_pmi_dict.get(token, None)
        if token_pmi is not None:
            attractor_token_ppmi_scores.append(max(0, token_pmi))

    attractor_lemma_freq_scores = list()
    attractor_lemma_ppmi_scores = list()
    for lemma in spacy_lemmas:
        lemma_freq = attractor_lemma_freq_dict.get(lemma, None)
        if lemma_freq is not None:
            attractor_lemma_freq_scores.append(sum(lemma_freq))
        lemma_pmi = attractor_lemma_pmi_dict.get(lemma, None)
        if lemma_pmi is not None:
            attractor_lemma_ppmi_scores.append(np.mean([max(0, lp) for lp in lemma_pmi]))

    return [attractor_token_freq_scores, attractor_token_ppmi_scores,
            attractor_lemma_freq_scores, attractor_lemma_ppmi_scores,
            len(spacy_tokens_lower), seed_parses]


def build_corpus(adversarial_samples_path, attractors_table_path, corpus_size, criterion,
                 max_homograph_count, constrain_sampling_region):
    """ Creates a WSD challenge corpus from the totality of the identified seed sentences. """

    assert criterion in ['seed_ppmi', 'adv_ppmi', 'ppmi_diff',
                         'seed_freq', 'adv_freq', 'freq_diff',
                         'length', 'random'], \
        '{:s} is not a valid selection criterion!'.format(criterion)

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
    seed_parses = dict()
    seed_score_cache = dict()
    adv_score_cache = dict()
    diff_score_cache = dict()
    seen_seeds = dict()
    seed_to_sample = dict()

    # Placeholders
    seed_attractor_token_freq_dict, seed_attractor_token_pmi_dict = None, None
    seed_attractor_lemma_freq_dict, seed_attractor_lemma_pmi_dict = None, None

    # Declare function used to compute sentence-level scores
    # sent_fun = np.mean
    sent_fun = sum

    # Pre-score seed sentences
    print('Looking up attractor scores ...')
    for term_id, term in enumerate(adversarial_samples_table.keys()):
        print('Looking-up the term \'{:s}\''.format(term))
        for seed_cluster in adversarial_samples_table[term].keys():

            if criterion in ['seed_ppmi', 'ppmi_diff', 'seed_freq', 'freq_diff']:
                # Compute sentence-level scores for the relevant cluster
                seed_sorted_attractor_freq = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY FREQ]']
                seed_sorted_attractor_pmi = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY PMI]']
                seed_attractor_token_freq_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_freq}
                seed_attractor_token_pmi_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_pmi}
                seed_attractor_lemma_freq_dict = dict()
                seed_attractor_lemma_pmi_dict = dict()
                for token in seed_attractor_token_freq_dict.keys():
                    attractor_lemma = attractors_table[term][seed_cluster]['[CONTEXT TOKENS]'][token]['[LEMMA]']
                    if seed_attractor_lemma_freq_dict.get(attractor_lemma, None) is None:
                        seed_attractor_lemma_freq_dict[attractor_lemma] = [seed_attractor_token_freq_dict[token]]
                        seed_attractor_lemma_pmi_dict[attractor_lemma] = [seed_attractor_token_pmi_dict[token]]
                    else:
                        seed_attractor_lemma_freq_dict[attractor_lemma].append(seed_attractor_token_freq_dict[token])
                        seed_attractor_lemma_pmi_dict[attractor_lemma].append(seed_attractor_token_pmi_dict[token])

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

                    if criterion in ['seed_freq', 'freq_diff']:
                        if seed_score_cache.get((seed_sentence, term, seed_cluster, adv_cluster), None) is None:
                            seed_scores = \
                                _get_sentence_scores(seed_sentence, seed_parses,
                                                     seed_attractor_token_freq_dict, seed_attractor_token_pmi_dict,
                                                     seed_attractor_lemma_freq_dict, seed_attractor_lemma_pmi_dict)
                            seed_parses = seed_scores[-1]
                            if len(seed_scores[0]) != 0:
                                seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = \
                                    sent_fun(seed_scores[0]) / seed_scores[4]
                            else:
                                seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = 0

                    if criterion in ['seed_ppmi', 'ppmi_diff']:
                        if seed_score_cache.get((seed_sentence, term, seed_cluster, adv_cluster), None) is None:
                            seed_scores = \
                                _get_sentence_scores(seed_sentence, seed_parses,
                                                     seed_attractor_token_freq_dict, seed_attractor_token_pmi_dict,
                                                     seed_attractor_lemma_freq_dict, seed_attractor_lemma_pmi_dict)
                            seed_parses = seed_scores[-1]
                            if len(seed_scores[1]) != 0:
                                seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = \
                                    sent_fun(seed_scores[1]) / seed_scores[4]
                            else:
                                seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = 0

                    # Find cluster with highest adversarial score
                    if criterion in ['adv_freq', 'adv_ppmi', 'freq_diff', 'ppmi_diff']:
                        if adv_score_cache.get((seed_sentence, term, seed_cluster, adv_cluster), None) is None:
                            top_adv_scores = None
                            # Iterate over sense clusters consistent with the mistranslation
                            for ac in attractors_table[term].keys():
                                if ac == seed_cluster:
                                    continue
                                # Compute sentence-level scores for the relevant cluster
                                adv_sorted_attractor_freq = \
                                    attractors_table[term][ac]['[SORTED ATTRACTORS BY FREQ]']
                                adv_sorted_attractor_pmi = \
                                    attractors_table[term][ac]['[SORTED ATTRACTORS BY PMI]']
                                adv_attractor_token_freq_dict = \
                                    {attr_tpl[0]: attr_tpl[1] for attr_tpl in adv_sorted_attractor_freq}
                                adv_attractor_token_pmi_dict = \
                                    {attr_tpl[0]: attr_tpl[1] for attr_tpl in adv_sorted_attractor_pmi}
                                adv_attractor_lemma_freq_dict, adv_attractor_lemma_pmi_dict = dict(), dict()
                                for token in adv_attractor_token_freq_dict.keys():
                                    attractor_lemma = \
                                        attractors_table[term][ac]['[CONTEXT TOKENS]'][token]['[LEMMA]']
                                    if adv_attractor_lemma_freq_dict.get(attractor_lemma, None) is None:
                                        adv_attractor_lemma_freq_dict[attractor_lemma] = \
                                            [adv_attractor_token_freq_dict[token]]
                                        adv_attractor_lemma_pmi_dict[attractor_lemma] = \
                                            [adv_attractor_token_pmi_dict[token]]
                                    else:
                                        adv_attractor_lemma_freq_dict[attractor_lemma] \
                                            .append(adv_attractor_token_freq_dict[token])
                                        adv_attractor_lemma_pmi_dict[attractor_lemma] \
                                            .append(adv_attractor_token_pmi_dict[token])

                                ac_scores = _get_sentence_scores(seed_sentence, seed_parses,
                                                                 adv_attractor_token_freq_dict,
                                                                 adv_attractor_token_pmi_dict,
                                                                 adv_attractor_lemma_freq_dict,
                                                                 adv_attractor_lemma_pmi_dict)

                                seed_parses = ac_scores[-1]
                                ac_scores = ac_scores[:-1]

                                if 'freq' in criterion:
                                    if top_adv_scores is None:
                                        top_adv_scores = (ac_scores[0], ac_scores[4])
                                    else:
                                        if sent_fun(ac_scores[0]) > sent_fun(top_adv_scores[0]):
                                            top_adv_scores = (ac_scores[0], ac_scores[4])
                                else:
                                    if top_adv_scores is None:
                                        top_adv_scores = (ac_scores[1], ac_scores[4])
                                    else:
                                        if sent_fun(ac_scores[1]) > sent_fun(top_adv_scores[0]):
                                            top_adv_scores = (ac_scores[1], ac_scores[4])

                            if len(top_adv_scores[0]) != 0:
                                adv_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = \
                                    sent_fun(top_adv_scores[0]) / top_adv_scores[1]
                            else:
                                adv_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = 0

                    if criterion in ['freq_diff', 'ppmi_diff']:
                        seed_score = seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)]
                        adv_score = adv_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)]
                        diff_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = adv_score - seed_score

                    if criterion == 'length':
                        seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = len(true_src_tokens)

                    if criterion == 'random':
                        # Assign the same score to every sentence for the random selection strategy
                        seed_score_cache[(seed_sentence, term, seed_cluster, adv_cluster)] = 0

                    seed_to_sample[(seed_sentence, term, seed_cluster, adv_cluster)] = sample

    if criterion in ['adv_freq', 'adv_ppmi']:
        score_cache = adv_score_cache
    elif criterion in ['freq_diff', 'ppmi_diff']:
        score_cache = diff_score_cache
    else:
        score_cache = seed_score_cache

    # Sort term samples by attractor property
    score_dist = list(score_cache.values())
    if criterion in ['seed_ppmi', 'seed_freq']:
        # min == possibly ambiguous / no known attractors
        # max == strong seed sense bias, hard to flip
        if constrain_sampling_region == 'True':
            sample_bound = np.quantile(score_dist, 0.1)
        else:
            sample_bound = 0.
    elif criterion in ['adv_ppmi', 'adv_freq']:
        # min == low bias towards the adversarial sense, possibly hard to flip
        # max == strong adversarial sense bias
        if constrain_sampling_region == 'True':
            sample_bound = np.quantile(score_dist, 0.9)
        else:
            sample_bound = max(score_dist)
    elif criterion in ['ppmi_diff', 'freq_diff']:
        # min: (high negative values) much stronger seed bias than adversarial bias
        # max: (low negative to positive values?) stronger adversarial bias than seed bias
        # sentences with PPMI of 0 are regarded as 'unbiased', which is a strong simplification
        if constrain_sampling_region == 'True':
            sample_bound = np.quantile(score_dist, 0.9)
        else:
            sample_bound = max(score_dist)
    else:
        sample_bound = 0

    # Sort term samples by attractor property
    print('Ranking and sampling adversarial samples ...')
    scored_seeds = [(k[0], k[1], k[2], k[3], v) for k, v in score_cache.items()]
    scored_seeds = sorted(scored_seeds, reverse=True, key=lambda x: len(x[0].split()))  # sort by length first
    if criterion in ['seed_ppmi', 'seed_freq']:
        sorted_seeds = sorted(scored_seeds, reverse=False, key=lambda x: x[-1])
    elif criterion in ['adv_ppmi', 'adv_freq', 'ppmi_diff', 'freq_diff', 'length']:
        sorted_seeds = sorted(scored_seeds, reverse=True, key=lambda x: x[-1])
    else:
        random.shuffle(scored_seeds)
        sorted_seeds = scored_seeds

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
        if criterion in ['seed_ppmi', 'seed_freq', 'length']:
            if ss[-1] < sample_bound:
                continue
        else:
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
        '_natural_challenge_{:s}_{:d}_{:d}_{:s}.json'\
        .format(criterion, corpus_size, max_homograph_count, constrain_sampling_region)
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
    parser.add_argument('--criterion', type=int, help='criterion for selecting sentences to be included in the challenge set',
                        required=True)
    parser.add_argument('--max_homograph_count', type=int, help='max samples per homograph',
                        required=True)
    parser.add_argument('--constrain_sampling_region', type=str,
                        help='whether to filter sentences based on their disambiguation bias')
    args = parser.parse_args()

    # Instantiate processing pipeline
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    STOP_WORDS = stopwords.words('english')

    build_corpus(args.adversarial_samples_path, args.attractors_table_path, args.corpus_size, args.criterion,
                 args.max_homograph_count, args.constrain_sampling_region)

