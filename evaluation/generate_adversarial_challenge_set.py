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


def build_corpus(adversarial_samples_path, attractors_table_path, corpus_size, adv_criterion, seed_criterion,
                 max_homograph_count, max_attr_combo_count, max_seed_combo_count):
    """ Creates a WSD challenge corpus from the totality of the generated adversarial samples. """

    assert adv_criterion in ['seed_ppmi', 'adv_ppmi', 'ppmi_diff',
                             'seed_freq', 'adv_freq', 'freq_diff',
                             'length', 'random', None], \
        '{} is not a valid selection criterion!'.format(seed_criterion)
    assert seed_criterion in ['seed_ppmi', 'adv_ppmi', 'ppmi_diff',
                              'seed_freq', 'adv_freq', 'freq_diff',
                              'length', 'random', None], \
        '{} is not a valid selection criterion!'.format(seed_criterion)

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
    seen_seeds = dict()
    seed_parses = dict()
    seed_score_cache = dict()
    adv_score_cache = dict()
    seed_sentence_scores = dict()
    seen_samples = dict()

    # Declare function used to compute sentence-level scores
    # sent_fun = np.mean
    sent_fun = sum

    # Pre-score seed sentences
    print('Scoring seed sentences ...')
    # Obtain scores based on seed sentence properties
    for term_id, term in enumerate(adversarial_samples_table.keys()):
        for seed_cluster in adversarial_samples_table[term].keys():
            # Compute sentence-level scores for the relevant cluster
            seed_sorted_attractor_freq = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY FREQ]']
            seed_sorted_attractor_pmi = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY PMI]']
            seed_attractor_token_freq_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_freq}
            seed_attractor_token_pmi_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_pmi}

            for adv_cluster in adversarial_samples_table[term][seed_cluster].keys():
                for sample in adversarial_samples_table[term][seed_cluster][adv_cluster]:
                    seed_sentence = sample[1].strip()

                    if seen_seeds.get((seed_sentence, term, seed_cluster), None) is not None:
                        continue
                    seen_seeds[(seed_sentence, term, seed_cluster)] = True

                    true_src_tokens = [tok.strip(punctuation_plus_space) for tok in seed_sentence.split()]
                    true_src_tokens = [tok for tok in true_src_tokens if len(tok) > 0]
                    # Ignore short sentences
                    if len(true_src_tokens) < 10:
                        continue

                    # Compute sentence-level scores
                    seed_scores = None
                    adv_scores = None
                    if 'seed' in seed_criterion or 'diff' in seed_criterion or seed_criterion == 'length':
                        seed_scores = seed_score_cache.get((seed_sentence, term, seed_cluster), None)
                        if seed_scores is None:
                            seed_scores = _get_sentence_scores(seed_sentence, seed_parses,
                                                               seed_attractor_token_freq_dict,
                                                               seed_attractor_token_pmi_dict)
                            seed_score_cache[(seed_sentence, term, seed_cluster)] = seed_scores[:-1]
                            seed_parses = seed_scores[-1]

                    if 'adv' in seed_criterion or 'diff' in seed_criterion:
                        adv_scores = adv_score_cache.get((seed_sentence, term, seed_cluster), None)
                        if adv_scores is None:
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
                                ac_scores = _get_sentence_scores(seed_sentence, seed_parses,
                                                                 adv_attractor_token_freq_dict,
                                                                 adv_attractor_token_pmi_dict)
                                seed_parses = ac_scores[-1]
                                ac_scores = ac_scores[:-1]

                                if 'freq' in seed_criterion:
                                    if top_adv_scores is None:
                                        top_adv_scores = (ac_scores[0], ac_scores)
                                    else:
                                        if sent_fun(ac_scores[0]) > sent_fun(top_adv_scores[0]):
                                            top_adv_scores = (ac_scores[0], ac_scores)
                                else:
                                    if top_adv_scores is None:
                                        top_adv_scores = (ac_scores[2], ac_scores)
                                    else:
                                        if sent_fun(ac_scores[2]) > sent_fun(top_adv_scores[0]):
                                            top_adv_scores = (ac_scores[2], ac_scores)

                            adv_scores = top_adv_scores[-1]
                            adv_score_cache[(seed_sentence, term, seed_cluster)] = adv_scores

                    # Isolate relevant scores
                    if seed_criterion == 'seed_freq':
                        sent_score = sent_fun(seed_scores[0]) / seed_scores[3]
                    elif seed_criterion == 'adv_freq':
                        sent_score = sent_fun(adv_scores[0]) / adv_scores[3]
                    elif seed_criterion == 'freq_diff':
                        sent_score = (sent_fun(adv_scores[0]) / adv_scores[3]) - \
                                     (sent_fun(seed_scores[0]) / seed_scores[3])
                    elif seed_criterion == 'seed_ppmi':
                        sent_score = sent_fun(seed_scores[2]) / seed_scores[3]
                    elif seed_criterion == 'adv_ppmi':
                        sent_score = sent_fun(adv_scores[2]) / adv_scores[3]
                    elif seed_criterion == 'ppmi_diff':
                        sent_score = (sent_fun(adv_scores[2]) / adv_scores[3]) - \
                                     (sent_fun(seed_scores[2]) / seed_scores[3])
                    elif seed_criterion == 'length':
                        sent_score = seed_scores[3]
                    else:
                        sent_score = random.random()

                    # Store scored sample
                    seed_sentence_scores[(seed_sentence, term, seed_cluster)] = sent_score

  
    # Initialize challenge set table
    all_samples = {'insert_at_homograph': list(),
                   'replace_at_homograph': list()}

    print('Ranking adversarial samples ...')
    if adv_criterion is not None:
        for term_id, term in enumerate(adversarial_samples_table.keys()):
            for seed_cluster in adversarial_samples_table[term].keys():
                # Compute sentence-level scores for the relevant cluster
                seed_sorted_attractor_freq = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY FREQ]']
                seed_sorted_attractor_pmi = attractors_table[term][seed_cluster]['[SORTED ATTRACTORS BY PMI]']
                seed_attractor_token_freq_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in
                                                  seed_sorted_attractor_freq}
                seed_attractor_token_pmi_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in seed_sorted_attractor_pmi}

                for adv_cluster in adversarial_samples_table[term][seed_cluster].keys():
                    # Compute sentence-level scores for the relevant cluster
                    adv_sorted_attractor_freq = attractors_table[term][adv_cluster]['[SORTED ATTRACTORS BY FREQ]']
                    adv_sorted_attractor_pmi = attractors_table[term][adv_cluster]['[SORTED ATTRACTORS BY PMI]']
                    adv_attractor_token_freq_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in
                                                     adv_sorted_attractor_freq}
                    adv_attractor_token_pmi_dict = {attr_tpl[0]: attr_tpl[1] for attr_tpl in
                                                    adv_sorted_attractor_pmi}

                    for sample in adversarial_samples_table[term][seed_cluster][adv_cluster]:
                        seed_sentence = sample[1].strip()
                        adv_sentence = sample[0].strip()
                        attractor = sample[3]
                        provenance_tag = sample[-3][-1]

                        # Ignore samples not generated with permitted strategies
                        if provenance_tag not in all_samples.keys():
                            continue

                        # Skip likely ambiguous sentences
                        seed_sentence_score = seed_sentence_scores.get((seed_sentence, term, seed_cluster), None)
                        if seed_sentence_score is None:
                            continue

                        # Ignore samples containing multiple instances of the attractor term
                        true_src_tokens = [tok.strip(punctuation_plus_space) for tok in seed_sentence.split()]
                        true_src_tokens = [tok for tok in true_src_tokens if len(tok) > 0]
                        adv_src_tokens = [tok.strip(punctuation_plus_space) for tok in adv_sentence.split()]
                        adv_src_tokens = [tok for tok in adv_src_tokens if len(tok) > 0]
                        if adv_src_tokens.count(attractor.strip(punctuation_plus_space)) > 1:
                            continue

                        # Ignore short sentences
                        if len(true_src_tokens) < 10:
                            continue

                        # Ignore samples with sentence-initial attractors
                        if 0 in sample[11]:
                            continue

                        # Skip duplicate adversarial samples
                        sample_key = (seed_sentence, adv_sentence, term, seed_cluster, adv_cluster, sample[7])
                        sample_seen = seen_samples.get(sample_key, False)
                        if sample_seen:
                            continue
                        else:
                            seen_samples[sample_key] = True

                        # Check if attractor was added by insertion or replacement
                        is_insert = 'insert' in provenance_tag
                        # Look up attractor scores
                        if is_insert:
                            sample_scores = sample[4]
                            seed_attr_freq = seed_attractor_token_freq_dict.get(attractor, 0)
                            adv_attr_freq = sample_scores['[SORTED ATTRACTORS BY FREQ]']
                            seed_attr_ppmi = max(0., seed_attractor_token_pmi_dict.get(attractor, 0.))
                            adv_attr_ppmi = max(0., sample_scores['[SORTED ATTRACTORS BY PMI]'])
                        else:
                            seed_attr_freq = 0
                            adv_attr_freq = 0
                            seed_attr_ppmi = 0
                            adv_attr_ppmi = 0

                        # Compute sentence-level scores
                        seed_scores = None
                        adv_scores = None
                        if 'seed' in adv_criterion or 'diff' in adv_criterion or adv_criterion == 'length':
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

                        if 'adv' in adv_criterion or 'diff' in adv_criterion:
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

                        # Isolate relevant scores
                        is_insert = int(is_insert)
                        if seed_scores is not None:
                            if len(seed_scores[0]) > 0:
                                seed_freq = sent_fun(seed_scores[0])
                                seed_ppmi = sent_fun(seed_scores[2])
                            else:
                                seed_freq = 0
                                seed_ppmi = 0

                        if adv_scores is not None:
                            if len(adv_scores[0]) > 0:
                                adv_freq = sent_fun(adv_scores[0])
                                adv_ppmi = sent_fun(adv_scores[2])
                            else:
                                adv_freq = 0
                                adv_ppmi = 0

                        if adv_criterion == 'seed_freq':
                            adv_sent_score = (seed_freq + seed_attr_freq) / (seed_scores[3] + is_insert)
                        elif adv_criterion == 'adv_freq':
                            adv_sent_score = (adv_freq + adv_attr_freq) / (adv_scores[3] + is_insert)
                        elif adv_criterion == 'freq_diff':
                            adv_sent_score = (adv_freq + adv_attr_freq) / (adv_scores[3] + is_insert) - \
                                         (seed_freq + seed_attr_freq) / (seed_scores[3] + is_insert)
                        elif adv_criterion == 'seed_ppmi':
                            adv_sent_score = (seed_ppmi + seed_attr_ppmi) / (seed_scores[3] + is_insert)
                        elif adv_criterion == 'adv_ppmi':
                            adv_sent_score = (adv_ppmi + adv_attr_ppmi) / (adv_scores[3] + is_insert)
                        elif adv_criterion == 'ppmi_diff':
                            adv_sent_score = (adv_ppmi + adv_attr_ppmi) / (adv_scores[3] + is_insert) - \
                                         (seed_ppmi + seed_attr_ppmi) / (seed_scores[3] + is_insert)
                        elif adv_criterion == 'length':
                            adv_sent_score = seed_scores[3]
                        else:
                            adv_sent_score = 0  # random sampling

                        # Store scored sample
                        all_samples[provenance_tag].\
                            append([seed_sentence_score, adv_sent_score, term, seed_cluster, adv_cluster, sample])


    print('Ranking and sampling adversarial samples ...')
    # Pool samples across generation strategies
    sample_pool = list()
    for key in all_samples.keys():
        for scored_sample in all_samples[key]:
            sample_pool.append(scored_sample + [key])
    # Sort according to measures
    if adv_criterion != 'random':
        sorted_by_adv = sorted(sample_pool, reverse='seed' not in seed_criterion, key=lambda x: x[0])
    else:
        sorted_by_adv = sample_pool
        random.shuffle(sorted_by_adv)
    if seed_criterion != 'random':
        sorted_by_seed = sorted(sample_pool, reverse='seed' not in seed_criterion, key=lambda x: x[0])
    else:
        sorted_by_seed = sample_pool
        random.shuffle(sorted_by_seed)

    # Rank by attractor property
    ranked_by_adv = dict()
    rank = 0
    last_score = None
    for tpl in sorted_by_adv:
        tpl_key = (tpl[5][0], tpl[5][7], tuple(tpl[5][11]), tpl[2], tpl[3], tpl[4])
        if tpl[1] != last_score:
            rank += 1
            last_score = tpl[1]
        if ranked_by_adv.get(tpl_key, None) is None:
            ranked_by_adv[tpl_key] = rank

    # Sort by mean rank, resolving ties by LM score
    updated_samples = list()
    for sample in sample_pool:
        sample_key = (sample[5][0], sample[5][7], tuple(sample[5][11]), sample[2], sample[3], sample[4])
        adv_rank = ranked_by_adv.get(sample_key, None)
        if adv_rank is None:
            continue
        joint_rank = adv_rank
        updated_samples.append((joint_rank, sample))
    sorted_samples = \
        [sample[1] for sample in sorted(updated_samples, reverse=False, key=lambda x: [x[0], float(x[1][5][16])])]

    # Sample samples
    samples_drawn = 0
    diff_terms = 0
    diff_seeds = 0
    diff_seed_clusters = 0
    diff_adv_clusters = 0

    diff_attractors = list()
    provenance_counts = dict()
    term_counts = dict()
    term_attractor_counts = dict()
    term_seed_counts = dict()
    unique_seeds = dict()

    for ss in sorted_samples:

        # Don't exceed specified challenge set size
        if samples_drawn == corpus_size:
            break

        # Ignore frequent terms
        term_count = term_counts.get(ss[2], 0)
        if term_count >= max_homograph_count > -1:
            continue

        # Ignore frequent term-attractor pairings
        attr_pair_count = term_attractor_counts.get((ss[2], ss[5][3]), 0)
        if attr_pair_count >= max_attr_combo_count > -1:
            continue

        # Ignore frequent seed-attractor pairings
        seed_pair_count = term_attractor_counts.get((ss[5][1].strip(), ss[5][3]), 0)
        if seed_pair_count >= max_seed_combo_count > -1:
            continue

        if corpus_samples_table.get(ss[2], None) is None:
            corpus_samples_table[ss[2]] = dict()
            diff_terms += 1

        if corpus_samples_table[ss[2]].get(ss[3], None) is None:
            corpus_samples_table[ss[2]][ss[3]] = dict()
            diff_seed_clusters += 1

        if corpus_samples_table[ss[2]][ss[3]].get(ss[4], None) is None:
            corpus_samples_table[ss[2]][ss[3]][ss[4]] = list()
            diff_adv_clusters += 1

        corpus_samples_table[ss[2]][ss[3]][ss[4]].append(ss[5])
        samples_drawn += 1

        # Track counts
        provenance_tag = ss[6]
        if provenance_counts.get(provenance_tag, None) is None:
            provenance_counts[provenance_tag] = 1
        else:
            provenance_counts[provenance_tag] += 1

        seed_key = (ss[5][1], ss[5][7])
        if unique_seeds.get(seed_key, None) is None:
            unique_seeds[seed_key] = True
            diff_seeds += 1

        if term_counts.get(ss[2], None) is None:
            term_counts[ss[2]] = 1
        else:
            term_counts[ss[2]] += 1

        if term_attractor_counts.get((ss[2], ss[5][3]), None) is None:
            term_attractor_counts[(ss[2], ss[5][3])] = 1
        else:
            term_attractor_counts[(ss[2], ss[5][3])] += 1

        if term_seed_counts.get((ss[2], ss[5][1].strip()), None) is None:
            term_seed_counts[(ss[2], ss[5][1].strip())] = 1
        else:
            term_seed_counts[(ss[2], ss[5][1].strip())] += 1

        diff_attractors.append((ss[2], ss[5][3]))

    diff_attractors = len(set(diff_attractors))

    print('FINAL STATS')
    min_samples = float('inf')
    max_samples = 0
    print('=' * 20)
    print('Evaluated {:d} terms, extracted {:d} samples'
          .format(len(adversarial_samples_table.keys()), samples_drawn))
    print('Total unique terms: {:d}'.format(diff_terms))
    print('Total unique seeds: {:d}'.format(diff_seeds))
    print('Total unique homograph-attractor pairings: {:d}'.format(diff_attractors))
    print('Mean unique seeds per term: {:.4f}'.format(diff_seeds / diff_terms))
    print('Mean unique seed clusters per term: {:.4f}'.format(diff_seed_clusters / diff_terms))
    print('Mean unique attractor clusters per term: {:.4f}'.format(diff_adv_clusters / diff_terms))
    print('Num samples per generation strategy:')
    for s, c in provenance_counts.items():
        print('{:s} : {:d}'.format(s, c))
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
        min_samples = term_entry_size if term_entry_size < min_samples else min_samples
        max_samples = term_entry_size if term_entry_size > max_samples else max_samples
    print('=' * 20)
    print('Min samples per term: {:d}'.format(min_samples))
    print('Max samples per term: {:d}'.format(max_samples))

    out_path = adversarial_samples_path[:-5] + '_two_step_V2_synthetic_challenge_{:s}_{:s}_{:d}_{:d}_{:d}.json'\
        .format(seed_criterion, adv_criterion, corpus_size, max_homograph_count, max_attr_combo_count)

    with open(out_path, 'w', encoding='utf8') as op:
        json.dump(corpus_samples_table, op, indent=3, sort_keys=True, ensure_ascii=False)
    print('Done!')
    print('Saved the challenge corpus table to {:s}'.format(out_path))


if __name__ == '__main__':
    # Instantiate processing pipeline
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    STOP_WORDS = stopwords.words('english')

    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_samples_path', type=str, help='path to the adversarial sample pool',
                        required=True)
    parser.add_argument('--attractors_table_path', type=str, help='path to the attractor pool',
                        required=True)
    parser.add_argument('--corpus_size', type=int, help='size of the challenge set in pairs',
                        required=True)
    parser.add_argument('--adv_criterion', type=str, help='criterion for the selection of attractors',
                        required=True)
    parser.add_argument('--seed_criterion', type=str, help='criterion for the selection of seed sentences',
                        required=True)
    parser.add_argument('--max_homograph_count', type=int, help='max samples per homograph')
    parser.add_argument('--max_attr_combo_count', type=int, help='max samples per homograph + attractor combination")
    parser.add_argument('--max_seed_combo_count', type=int, help='max samples per homograph + seed sentence combination')
    args = parser.parse_args()

    build_corpus(args.adversarial_samples_path, args.attractors_table_path, args.corpus_size, args.adv_criterion, args.seed_criterion,
                 args.max_homograph_count, args.max_attr_combo_count, args.max_seed_combo_count)

