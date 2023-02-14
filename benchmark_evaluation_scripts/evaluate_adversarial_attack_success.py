# Strategy
# 1. Separate examples into successful and unsuccessful ones
#    For now, success if a flip to the adversarial cluster, failure is translation consistent wth seed cluster
# 2. For evaluation, consider:
#    a. attractor scores
#    b. seed scores according to modifiers
#    c. three-way interaction between attractor scores, seed scores, and attack success
#    d. Distance between attractor and target sense alternatives with the model's embedding space

import re
import os
import sys
import json
import spacy
import string
import logging
import argparse

from nltk.corpus import stopwords

ATTN_WINDOW_RANGE = 0
GEN_METHODS = ['insert_at_homograph', 'replace_at_homograph', 'insert_at_other', 'replace_at_other']


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


# TODO / NOTE: Only compatible with word-level attractors, for now
def _get_nmt_label(translation,
                   adv_src,
                   true_src,
                   true_tgt,
                   ambiguous_token,
                   ambiguous_token_loc_seed,
                   ambiguous_token_loc_adv,
                   attractor_tokens_loc,
                   seed_cluster_id,
                   adv_cluster_id,
                   other_cluster_ids,
                   sense_lemmas_to_cluster,
                   cluster_to_sense_lemmas,
                   sense_tokens_to_cluster,
                   cluster_to_sense_tokens,
                   alignments,
                   is_adv,
                   true_translation=None,
                   true_alignments=None):

    """ Helper function for evaluating whether the translations of true and adversarially perturbed source samples
    perform lexical WSD correctly. """

    # Lemmatize the translation for higher coverage of attack successes (preserve punctuation)
    _, spacy_tokens_lower, spacy_lemmas, _, ws_tokens, ws_tokens_lower, ws_to_spacy_map, spacy_to_ws_map = \
        _process_strings(translation,
                         tgt_nlp,
                         get_lemmas=True,
                         get_pos=False,
                         remove_stopwords=False,
                         replace_stopwords=False,
                         get_maps=True)

    # All sequences are Moses-tokenized and can be split at whitespace
    adv_src_tokens = adv_src.strip().split()
    true_src_tokens = true_src.strip().split()
    ambiguous_token_loc_seed = ambiguous_token_loc_seed[0]

    # Check that the provided locations of the ambiguous token are correct
    ambiguous_token_in_seed = true_src_tokens[ambiguous_token_loc_seed].lower().strip(punctuation_plus_space)

    assert ambiguous_token.lower().strip(punctuation_plus_space) == \
        ambiguous_token_in_seed or ambiguous_token[:-1] in ambiguous_token_in_seed, \
        'Ambiguous token \'{:s}\' does not match the true source token \'{:s}\' at the token location'\
        .format(ambiguous_token, ambiguous_token_in_seed)
    ambiguous_token_in_adv = adv_src_tokens[ambiguous_token_loc_adv].lower().strip(punctuation_plus_space)
    assert ambiguous_token.lower().strip(punctuation_plus_space) == \
        ambiguous_token_in_adv or ambiguous_token[:-1] in ambiguous_token_in_adv, \
        'Ambiguous token \'{:s}\' does not match the adversarial source token \'{:s}\' at the token location' \
        .format(ambiguous_token, ambiguous_token_in_adv)

    ambiguous_token_loc = ambiguous_token_loc_adv if is_adv else ambiguous_token_loc_seed

    other_cluster_lemmas = list()
    for cluster_id in other_cluster_ids:
        other_cluster_lemmas += cluster_to_sense_lemmas[cluster_id]
    other_cluster_tokens = list()
    for cluster_id in other_cluster_ids:
        other_cluster_tokens += cluster_to_sense_tokens[cluster_id]

    target_hits = list()
    attractor_term_translated = False
    # If alignments are available, look-up target sense aligned with the ambiguous source term
    alignments = alignments.strip()
    # Convert alignments into a more practical format, first
    line_align_table = dict()
    for word_pair in alignments.strip().split():
        src_id, tgt_id = word_pair.split('-')
        src_id = int(src_id)
        tgt_id = int(tgt_id)
        if not line_align_table.get(src_id, None):
            line_align_table[src_id] = [tgt_id]
        else:
            line_align_table[src_id].append(tgt_id)

    if ambiguous_token_loc in line_align_table.keys():
        homograph_aligned = True
        # Extend alignment window
        line_align_table_entry = sorted(line_align_table[ambiguous_token_loc])
        min_tgt_idx = max(0, line_align_table_entry[0] - ATTN_WINDOW_RANGE)
        max_tgt_idx = min(len(spacy_lemmas) - 1, line_align_table_entry[-1] + ATTN_WINDOW_RANGE)
        # Map from target whitespace to target spacy tokens
        min_tgt_idx = ws_to_spacy_map[min_tgt_idx][0]
        max_tgt_idx = ws_to_spacy_map[max_tgt_idx][0]
        tgt_window = range(min_tgt_idx, max_tgt_idx)
        # Check if aligned translation lemmas include correct / flipped source term translations
        aligned_translation_lemmas = [spacy_lemmas[idx] for idx in tgt_window]
        aligned_tgt_tokens = [spacy_tokens_lower[idx] for idx in tgt_window]

        for atl_id, atl in enumerate(aligned_translation_lemmas):
            # Skip stopwords
            if atl == '<STPWRD>':
                continue
            # Strip leading and trailing punctuation
            atl = atl.strip(punctuation_plus_space)
            # Match lemmas
            if sense_lemmas_to_cluster is not None:
                if sense_lemmas_to_cluster.get(atl, None):
                    target_hits.append(sense_lemmas_to_cluster[atl])
                else:
                    maybe_lemma_matches = sorted(cluster_to_sense_lemmas[seed_cluster_id] +
                                                 cluster_to_sense_lemmas[adv_cluster_id] +
                                                 other_cluster_lemmas, reverse=True, key=lambda x: len(x))

                    for maybe_lemma_match in maybe_lemma_matches:
                        if atl.startswith(maybe_lemma_match) or atl.endswith(maybe_lemma_match) or \
                                atl[:-1].endswith(maybe_lemma_match):
                            target_hits.append(sense_lemmas_to_cluster[maybe_lemma_match])
                            break
            # Match tokens
            if len(target_hits) == 0 and sense_tokens_to_cluster is not None:
                att = aligned_tgt_tokens[atl_id].strip(punctuation_plus_space)
                if sense_tokens_to_cluster.get(att, None):
                    target_hits.append(sense_tokens_to_cluster[att])
                else:
                    maybe_token_matches = \
                        sorted(cluster_to_sense_tokens[seed_cluster_id] +
                               cluster_to_sense_tokens[adv_cluster_id] +
                               other_cluster_tokens, reverse=True, key=lambda x: len(x))

                    for maybe_token_match in maybe_token_matches:
                        if att.startswith(maybe_token_match) or att.endswith(maybe_token_match.lower()) or \
                                att[:-1].endswith(maybe_token_match.lower()):
                            target_hits.append(sense_tokens_to_cluster[maybe_token_match.lower()])
                            break
    else:
        homograph_aligned = False

    # TODO: Can / should this be improved?
    # Check if the attractor term(s) have been translated (assumes translation if alignment was found)
    # NOTE: Currently works ONLY FOR 1-WORD ATTRACTORS (e.g. adjectives)
    is_compound = False
    if is_adv:
        if line_align_table.get(attractor_tokens_loc[0], None) is not None:
            attractor_term_translated = True
        else:
            # Check if true and adversarial translations are identical
            if translation.strip() == true_translation.strip():
                attractor_term_translated = False
            # Try to check whether the attractor has been translated as part of a compound
            else:

                # TODO: DEBUGGING
                # print('=== Attractor not aligned ===')
                # print(true_src.strip())
                # print(adv_src.strip())
                # print(true_translation.strip())
                # print(true_alignments.strip())
                # print(translation.strip())
                # print(alignments)
                # print(attractor_tokens_loc)

                # Look up alignments for the seed translation
                true_line_align_table = dict()
                for word_pair in true_alignments.strip().split():
                    src_id, tgt_id = word_pair.split('-')
                    src_id = int(src_id)
                    tgt_id = int(tgt_id)
                    if not true_line_align_table.get(src_id, None):
                        true_line_align_table[src_id] = [tgt_id]
                    else:
                        true_line_align_table[src_id].append(tgt_id)

                # Check if the modified noun is aligned to the same position in both translations
                modified_align_true = true_line_align_table.get(attractor_tokens_loc[0], [])
                modified_align_adv = line_align_table.get(attractor_tokens_loc[0] + 1, [])

                # TODO: DEBUGGING
                # print(modified_align_true)
                # print(modified_align_adv)

                true_translation_tokens = true_translation.strip().lower().split()
                aligned_true_tokens = [true_translation_tokens[true_loc].strip(punctuation_plus_space)
                                       for true_loc in modified_align_true]
                aligned_adv_tokens = [ws_tokens_lower[adv_loc].strip(punctuation_plus_space)
                                      for adv_loc in modified_align_adv]
                aligned_token_overlap = set(aligned_true_tokens) & set(aligned_adv_tokens)

                for true_token in aligned_true_tokens:
                    for adv_token in aligned_adv_tokens:
                        if true_token in aligned_token_overlap or adv_token in aligned_token_overlap:
                            continue
                        else:
                            if true_token != adv_token:
                                if true_token in adv_token or len(adv_token) > len(true_token) + 3 or \
                                        (adv_token not in true_token and true_token[-3:] == adv_token[-3:]):
                                    is_compound = True
                                    break
                if is_compound:
                    # Assume attractor and modified term have been jointly translated into a target compound
                    attractor_term_translated = True

            # TODO: DEBUGGING
            # print('ATTRACTOR TRANSLATED: {}'.format(attractor_term_translated))
            # print('\n')

    # If no alignments are available, match the known target sense
    if len(target_hits) == 0:
        for stl_id, stl in enumerate(spacy_lemmas):
            # if stl_id not in range(ambiguous_token_spacy_loc_seed - 3, ambiguous_token_spacy_loc_seed + 4):
            #     continue
            # Skip stopwords
            if stl == '<STPWRD>':
                continue
            # Strip leading and trailing punctuation
            stl = stl.strip(punctuation_plus_space)
            # Match lemmas
            if sense_lemmas_to_cluster is not None:
                if sense_lemmas_to_cluster.get(stl, None):
                    target_hits.append(sense_lemmas_to_cluster[stl])
                else:
                    maybe_lemma_matches = \
                        sorted(cluster_to_sense_lemmas[seed_cluster_id] +
                               cluster_to_sense_lemmas[adv_cluster_id] +
                               other_cluster_lemmas, reverse=True, key=lambda x: len(x))
                    for maybe_lemma_match in maybe_lemma_matches:
                        if stl.startswith(maybe_lemma_match) or stl.endswith(maybe_lemma_match) or \
                                stl[:-1].endswith(maybe_lemma_match):
                            target_hits.append(sense_lemmas_to_cluster[maybe_lemma_match])
                            break
            # Match tokens
            if len(target_hits) == 0 and sense_tokens_to_cluster is not None:
                stt = spacy_tokens_lower[stl_id].strip(punctuation_plus_space)
                if sense_tokens_to_cluster.get(stt, None):
                    target_hits.append(sense_tokens_to_cluster[stt])
                else:
                    maybe_token_matches = \
                        sorted(cluster_to_sense_tokens[seed_cluster_id] +
                               cluster_to_sense_tokens[adv_cluster_id] +
                               other_cluster_tokens, reverse=True, key=lambda x: len(x))
                    for maybe_token_match in maybe_token_matches:
                        if stt.startswith(maybe_token_match) or stt.endswith(maybe_token_match.lower()) or \
                                stt[:-1].endswith(maybe_token_match.lower()):
                            try:
                                target_hits.append(sense_tokens_to_cluster[maybe_token_match.lower()])
                                break
                            except KeyError:
                                pass

    # Source homograph is assumed to be translated if:
    # 1. Homograph is aligned
    # 2. Homograph is not aligned, but len(target_hits) > 0
    # 3. Homograph is not aligned and len(target_hits) == 0,
    #    but attractor modifies homograph and is translated into a compound
    if homograph_aligned:
        homograph_translated = True
    else:
        if len(target_hits) > 0:
            homograph_translated = True
        else:
            if is_adv:
                if is_compound and ambiguous_token_loc == (attractor_tokens_loc[0] + 1):
                    homograph_translated = True
                else:
                    homograph_translated = False
            else:
                homograph_translated = False

    # Flatten target hits
    target_hits = [hit[1] for hit_list in target_hits for hit in hit_list]
    # If target term is ambiguous, assume the translation is correct
    if seed_cluster_id in target_hits:

        # TODO: DEBUGGING
        # logging.info('-' * 10)
        # logging.info('NOT FLIPPED')
        # logging.info('IS ADV: {}'.format(pair_is_adv))
        # logging.info(true_src.strip())
        # logging.info(adv_src)
        # logging.info(translation)

        return 'not_flipped', target_hits, attractor_term_translated

    elif adv_cluster_id in target_hits:

        # TODO: DEBUGGING
        # logging.info('-' * 10)
        # logging.info('FLIPPED TO ATTR')
        # logging.info('IS ADV: {}'.format(is_adv))
        # logging.info(true_src.strip())
        # logging.info(adv_src)
        # logging.info(translation)

        return 'flipped_to_attr', target_hits, attractor_term_translated

    elif len(set(other_cluster_ids) & set(target_hits)) >= 1:

        # TODO: DEBUGGING
        # logging.info('-' * 10)
        # logging.info('FLIPPED TO OTHER')
        # logging.info('IS ADV: {}'.format(pair_is_adv))
        # logging.info(true_src.strip())
        # logging.info(adv_src)
        # logging.info(translation)

        return 'flipped_to_other', target_hits, attractor_term_translated

    # i.e. target_hits is empty
    else:
        if homograph_translated:

            # TODO: DEBUGGING
            # logging.info('-' * 10)
            # logging.info('MAYBE FLIPPED')
            # logging.info('IS ADV: {}'.format(pair_is_adv))
            # logging.info(true_src.strip())
            # logging.info(adv_src)
            # logging.info(translation)

            return 'maybe_flipped', target_hits, attractor_term_translated

        else:
            # TODO: DEBUGGING
            # if homograph_translated:
            #     print('+' * 20)
            #     print('HOMOGRAPH IS TRANSLATED')
            # else:
            #     print('-' * 20)
            #     print('HOMOGRAPH IS NOT TRANSLATED')
            #
            # print(is_adv)
            # print(homograph_aligned)
            # print(ambiguous_token_loc)
            # print(line_align_table)
            # print(true_src.strip())
            # print(true_translation.strip() if true_translation is not None else None)
            # print(true_alignments.strip() if true_alignments is not None else None)
            # print(adv_src.strip())
            # print(translation.strip())
            # print(alignments.strip())

            return 'deleted_homograph', target_hits, attractor_term_translated


def _build_cluster_lookup(sense_clusters_table):

    """ Post-processes the scraped target sense cluster table by constructing a sense-to-cluster_id lookup table """

    # Initialize empty tables
    logging.info('Constructing the cluster lookup table ...')
    sense_to_cluster_table = dict()

    # Fill tables
    for src_term in sense_clusters_table.keys():
        logging.info('Looking-up the term \'{:s}\''.format(src_term))
        sense_to_cluster_table[src_term] = dict()
        for cluster_id in sense_clusters_table[src_term].keys():
            # Construct cluster-ID lookup table entry
            for tgt_sense in sense_clusters_table[src_term][cluster_id]['[SENSES]']:
                # Lemmatizing single words is not ideal, but is expected to improve attractor recall
                _, _, tgt_lemmas, _, _, _, _, _ = \
                    _process_strings(tgt_sense, tgt_nlp, True, False, False, False, False)
                # Multi-word targets are excluded for simplicity (as a result, some words are dropped)
                if len(tgt_lemmas) < 1:
                    continue
                tgt_lemma = tgt_lemmas[0]
                if len(tgt_lemma) > 0:
                    if not sense_to_cluster_table[src_term].get(tgt_lemma, None):
                        sense_to_cluster_table[src_term][tgt_lemma] = [(tgt_sense, cluster_id, True)]
                    else:
                        sense_to_cluster_table[src_term][tgt_lemma].append((tgt_sense, cluster_id, True))

            # Check for blacklisted and ambiguous senses
            senses_to_ignore = list()
            senses_to_ignore += sense_clusters_table[src_term][cluster_id].get('[BLACKLISTED SENSES]', [])
            senses_to_ignore += sense_clusters_table[src_term][cluster_id].get('[AMBIGUOUS SENSES]', [])
            for tgt_sense in senses_to_ignore:
                # Lemmatizing single words is not ideal, but is expected to improve attractor recall
                _, _, tgt_lemmas, _, _, _, _, _ = \
                    _process_strings(tgt_sense, tgt_nlp, True, False, False, False, False)
                # Multi-word targets are excluded for simplicity (as a result, some words are dropped)
                if len(tgt_lemmas) < 1:
                    continue
                tgt_lemma = tgt_lemmas[0]
                if len(tgt_lemma) > 0:
                    if not sense_to_cluster_table[src_term].get(tgt_lemma, None):
                        sense_to_cluster_table[src_term][tgt_lemma] = [(tgt_sense, cluster_id, False)]
                    else:
                        sense_to_cluster_table[src_term][tgt_lemma].append((tgt_sense, cluster_id, False))
    return sense_to_cluster_table


def evaluate_attack_success(adversarial_samples_path,
                            true_sources_path,
                            adversarial_sources_path,
                            true_translations_path,
                            adversarial_translations_path,
                            true_alignments_path,
                            adversarial_alignments_path,
                            attractors_path,
                            sense_clusters_path,
                            output_tables_dir):

    """ Detects successful attacks and computes correlation between attack success and various metrics. """

    def _score_and_filter(combined_sample, attractors_entry, ambiguous_term):
        """ Helper function for filtering a single adversarial sample. """

        # Combined sample contents:

        # [true_sources[line_id],
        # true_translations[line_id],
        # adv_translations[line_id],
        # true_alignments[line_id],
        # adversarial_alignments[line_id],
        # adv_to_sample[adv_sources_line.strip()][0] - challenge sample,
        # adv_to_sample[adv_sources_line.strip()][1] - term]

        sense_lemmas_to_cluster = sense_lemmas_to_cluster_table.get(ambiguous_term, None)
        sense_tokens_to_cluster = sense_tokens_to_cluster_table.get(ambiguous_term, None)
        cluster_to_sense_lemmas = cluster_to_sense_lemmas_table.get(ambiguous_term, None)
        cluster_to_sense_tokens = cluster_to_sense_tokens_table.get(ambiguous_term, None)

        # Unpack
        true_translation = combined_sample[1]
        adv_translation = combined_sample[2]
        true_line_alignments = combined_sample[3]
        adv_line_alignments = combined_sample[4]
        challenge_entry = combined_sample[5]

        adv_src = challenge_entry[0]
        true_src = challenge_entry[1]
        true_tgt = challenge_entry[2]
        attractor = challenge_entry[3]
        attractor_scores = challenge_entry[4]
        ambiguous_token_ws_loc_adv = challenge_entry[5]
        ambiguous_token_loc_seed = challenge_entry[7]
        ambiguous_token_ws_loc_seed = challenge_entry[8]  # list
        tgt_sense_ws_loc_seed = challenge_entry[10]  # list
        attractor_tokens_ws_loc = challenge_entry[11]  # list
        seed_cluster_id = challenge_entry[12]
        adv_cluster_id = challenge_entry[13]
        seed_cluster_senses = challenge_entry[14]
        adv_cluster_senses = challenge_entry[15]
        lm_score_ratio = float(challenge_entry[16])
        provenance_tags = challenge_entry[17]

        # Ignore samples containing multiple instances of the attractor term
        true_src_tokens = [tok.strip(punctuation_plus_space) for tok in true_src.split()]
        true_src_tokens = [tok for tok in true_src_tokens if len(tok) > 0]
        adv_src_tokens = [tok.strip(punctuation_plus_space) for tok in adv_src.split()]
        adv_src_tokens = [tok for tok in adv_src_tokens if len(tok) > 0]
        if adv_src_tokens.count(attractor.strip(punctuation_plus_space)) > 1:
            return None
        # Ignore samples with sentence-initial attractors
        if 0 in attractor_tokens_ws_loc:
            return None
        # Ignore short sentences
        if len(true_src_tokens) < 10:
            return None

        other_cluster_ids = list(attractors_entry.keys())
        other_cluster_ids.pop(other_cluster_ids.index(seed_cluster_id))
        try:
            other_cluster_ids.pop(other_cluster_ids.index(adv_cluster_id))
        except ValueError:
            # TODO / NOTE: Needed for evaluation of negative samples
            pass

        # Get NMT labels
        true_nmt_label, true_translation_sense_clusters, _ = \
            _get_nmt_label(true_translation,
                           adv_src,
                           true_src,
                           true_tgt,
                           ambiguous_term,
                           ambiguous_token_ws_loc_seed,
                           ambiguous_token_ws_loc_adv,
                           attractor_tokens_ws_loc,
                           seed_cluster_id,
                           adv_cluster_id,
                           other_cluster_ids,
                           sense_lemmas_to_cluster,
                           cluster_to_sense_lemmas,
                           sense_tokens_to_cluster,
                           cluster_to_sense_tokens,
                           true_line_alignments,
                           is_adv=False)

        adv_nmt_label, adv_translation_sense_clusters, attractor_is_translated = \
            _get_nmt_label(adv_translation,
                           adv_src,
                           true_src,
                           true_tgt,
                           ambiguous_term,
                           ambiguous_token_ws_loc_seed,
                           ambiguous_token_ws_loc_adv,
                           attractor_tokens_ws_loc,
                           seed_cluster_id,
                           adv_cluster_id,
                           other_cluster_ids,
                           sense_lemmas_to_cluster,
                           cluster_to_sense_lemmas,
                           sense_tokens_to_cluster,
                           cluster_to_sense_tokens,
                           adv_line_alignments,
                           is_adv=True,
                           true_translation=true_translation,
                           true_alignments=true_line_alignments)

        # Assemble table entry
        new_table_entry = [adv_src,
                           adv_translation,
                           true_src,
                           true_translation,
                           true_tgt,
                           ambiguous_term,
                           attractor,
                           attractor_scores,
                           ambiguous_token_ws_loc_adv,
                           ambiguous_token_ws_loc_seed,
                           tgt_sense_ws_loc_seed,
                           attractor_tokens_ws_loc,
                           seed_cluster_id,
                           adv_cluster_id,
                           seed_cluster_senses,
                           adv_cluster_senses,
                           adv_translation_sense_clusters,
                           attractor_is_translated,
                           lm_score_ratio,
                           provenance_tags,
                           (true_nmt_label, adv_nmt_label)]

        # Sort true samples into appropriate output tables, based on filtering outcome
        seed_table_to_expand = None
        if true_nmt_label == 'not_flipped':
            stat_dict['num_adversarial_samples'][0] += 1
            if not unique_samples.get((true_src, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_good_translations'] += 1
                unique_samples[(true_src, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_good_translations
            label_id = 0
        elif true_nmt_label == 'flipped_to_attr':
            stat_dict['num_adversarial_samples'][1] += 1
            if not unique_samples.get((true_src, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_bad_translations'] += 1
                unique_samples[(true_src, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_bad_translations
            label_id = 1
        elif true_nmt_label == 'flipped_to_other':
            stat_dict['num_adversarial_samples'][2] += 1
            if not unique_samples.get((true_src, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_bad_translations'] += 1
                unique_samples[(true_src, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_bad_translations
            label_id = 2
        elif true_nmt_label == 'deleted_homograph':
            stat_dict['num_adversarial_samples'][3] += 1
            if not unique_samples.get((true_src, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_deleted_homograph'] += 1
                unique_samples[(true_src, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_deleted_homograph
            label_id = 3
        else:
            stat_dict['num_adversarial_samples'][4] += 1
            if not unique_samples.get((true_src, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_maybe_bad_translations'] += 1
                unique_samples[(true_src, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_maybe_bad_translations
            label_id = 4

        # Sort adversarial samples
        if adv_nmt_label == 'flipped_to_attr':
            stat_dict['num_flipped_to_attr_sense'][label_id] += 1
            table_to_expand = flipped_to_attr_sense_adv_samples
            if attractor_is_translated:
                attractor_deletion_table = flipped_to_attr_sense_adv_samples_kept_attr
                stat_dict['num_flipped_to_attr_sense_kept_attr'][label_id] += 1
            else:
                attractor_deletion_table = flipped_to_attr_sense_adv_samples_deleted_attr
                stat_dict['num_flipped_to_attr_sense_deleted_attr'][label_id] += 1
        elif adv_nmt_label == 'flipped_to_other':
            stat_dict['num_flipped_to_other_sense'][label_id] += 1
            table_to_expand = flipped_to_other_sense_adv_samples
            if attractor_is_translated:
                attractor_deletion_table = flipped_to_other_sense_adv_samples_kept_attr
                stat_dict['num_flipped_to_other_sense_kept_attr'][label_id] += 1
            else:
                attractor_deletion_table = flipped_to_other_sense_adv_samples_deleted_attr
                stat_dict['num_flipped_to_other_sense_deleted_attr'][label_id] += 1
        elif adv_nmt_label == 'deleted_homograph':
            stat_dict['num_deleted_homograph'][label_id] += 1
            table_to_expand = deleted_homograph_adv_samples
            if attractor_is_translated:
                attractor_deletion_table = deleted_homograph_adv_samples_kept_attr
                stat_dict['num_deleted_homograph_kept_attr'][label_id] += 1
            else:
                attractor_deletion_table = deleted_homograph_adv_samples_deleted_attr
                stat_dict['num_deleted_homograph_deleted_attr'][label_id] += 1
        elif adv_nmt_label == 'maybe_flipped':
            stat_dict['num_maybe_flipped'][label_id] += 1
            table_to_expand = maybe_flipped_adv_samples
            if attractor_is_translated:
                attractor_deletion_table = maybe_flipped_adv_samples_kept_attr
                stat_dict['num_maybe_flipped_kept_attr'][label_id] += 1
            else:
                attractor_deletion_table = maybe_flipped_adv_samples_deleted_attr
                stat_dict['num_maybe_flipped_deleted_attr'][label_id] += 1
        else:
            stat_dict['num_not_flipped'][label_id] += 1
            table_to_expand = not_flipped_adv_samples
            if attractor_is_translated:
                attractor_deletion_table = not_flipped_adv_samples_kept_attr
                stat_dict['num_not_flipped_kept_attr'][label_id] += 1
            else:
                attractor_deletion_table = not_flipped_adv_samples_deleted_attr
                stat_dict['num_not_flipped_deleted_attr'][label_id] += 1

        # Collect seed translations
        if seed_table_to_expand is not None:
            if not seed_table_to_expand.get(ambiguous_term, None):
                seed_table_to_expand[ambiguous_term] = dict()
            if not seed_table_to_expand[ambiguous_term].get(seed_cluster_id, None):
                seed_table_to_expand[ambiguous_term][seed_cluster_id] = dict()
            if not seed_table_to_expand[ambiguous_term][seed_cluster_id].get(adv_cluster_id, None):
                seed_table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id] = dict()
            if not seed_table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id].get(true_src, None):
                seed_table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id][true_src] = list()
            seed_table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id][true_src]\
                .append([true_translation, true_tgt, true_translation_sense_clusters, ambiguous_token_loc_seed])

        # Collect attack success samples
        if not table_to_expand.get(ambiguous_term, None):
            table_to_expand[ambiguous_term] = dict()
        if not table_to_expand[ambiguous_term].get(seed_cluster_id, None):
            table_to_expand[ambiguous_term][seed_cluster_id] = dict()
        if not table_to_expand[ambiguous_term][seed_cluster_id].get(adv_cluster_id, None):
            table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id] = dict()
        if not table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id].get(true_src, None):
            table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id][true_src] = list()
        table_to_expand[ambiguous_term][seed_cluster_id][adv_cluster_id][true_src].append(new_table_entry)

        # Collect attractor deletion / retention samples
        single_attractor_deletion_table = attractor_deletion_table
        if attractor_is_translated:
            joined_attractor_deletion_table = all_adv_samples_kept_attractor
        else:
            joined_attractor_deletion_table = all_adv_samples_deleted_attractor
        for attractor_deletion_table in [single_attractor_deletion_table, joined_attractor_deletion_table]:
            if not attractor_deletion_table.get(ambiguous_term, None):
                attractor_deletion_table[ambiguous_term] = dict()
            if not attractor_deletion_table[ambiguous_term].get(seed_cluster_id, None):
                attractor_deletion_table[ambiguous_term][seed_cluster_id] = dict()
            if not attractor_deletion_table[ambiguous_term][seed_cluster_id].get(adv_cluster_id, None):
                attractor_deletion_table[ambiguous_term][seed_cluster_id][adv_cluster_id] = dict()
            if not attractor_deletion_table[ambiguous_term][seed_cluster_id][adv_cluster_id].get(true_src, None):
                attractor_deletion_table[ambiguous_term][seed_cluster_id][adv_cluster_id][true_src] = list()
            attractor_deletion_table[ambiguous_term][seed_cluster_id][adv_cluster_id][true_src]\
                .append(new_table_entry)

        # Track effectiveness of sample generation strategies
        method_id = GEN_METHODS.index(provenance_tags[-1])
        stat_dict['all_samples_method_count'][method_id] += 1

        if label_id == 0:
            stat_dict['good_seed_translations_method_count'][method_id] += 1
            if adv_nmt_label == 'flipped_to_attr':
                stat_dict['flipped_to_attr_sense_method_count_good_seed'][method_id] += 1
            elif adv_nmt_label == 'deleted_homograph':
                stat_dict['deleted_homograph_method_count_good_seed'][method_id] += 1

        elif label_id == 1:
            stat_dict['attr_sense_seed_translations_method_count'][method_id] += 1
            if adv_nmt_label == 'deleted_homograph':
                stat_dict['deleted_homograph_method_count_attr_seed'][method_id] += 1

        elif label_id == 2:
            stat_dict['other_sense_seed_translations_method_count'][method_id] += 1
            if adv_nmt_label == 'flipped_to_attr':
                stat_dict['flipped_to_attr_sense_method_count_other_seed'][method_id] += 1
            elif adv_nmt_label == 'deleted_homograph':
                stat_dict['deleted_homograph_method_count_other_seed'][method_id] += 1

        if not attractor_is_translated:
            stat_dict['deleted_attr_method_count'][method_id] += 1

    def _show_stats():
        """ Helper for reporting on the generation process. """

        def _pc(enum, denom):
            """ Helper function for computing percentage of the evaluated sample type """
            return (enum / denom) * 100 if denom > 0 else 0

        logging.info('-' * 20)

        num_all_seed_samples = \
            stat_dict['num_true_samples_good_translations'] + \
            stat_dict['num_true_samples_bad_translations'] + \
            stat_dict['num_true_samples_deleted_homograph'] + \
            stat_dict['num_true_samples_maybe_bad_translations']
        num_all_incorrect_adv_translations = \
            sum(stat_dict['num_flipped_to_attr_sense']) + \
            sum(stat_dict['num_flipped_to_other_sense']) + \
            sum(stat_dict['num_deleted_homograph']) + \
            sum(stat_dict['num_maybe_flipped'])
        num_all_deleted_attractor = \
            sum(stat_dict['num_flipped_to_attr_sense_deleted_attr']) + \
            sum(stat_dict['num_flipped_to_other_sense_deleted_attr']) + \
            sum(stat_dict['num_deleted_homograph_deleted_attr']) + \
            sum(stat_dict['num_maybe_flipped_deleted_attr']) + \
            sum(stat_dict['num_not_flipped_deleted_attr'])
        num_all_kept_attractor = \
            sum(stat_dict['num_flipped_to_attr_sense_kept_attr']) + \
            sum(stat_dict['num_flipped_to_other_sense_kept_attr']) + \
            sum(stat_dict['num_deleted_homograph_kept_attr']) + \
            sum(stat_dict['num_maybe_flipped_kept_attr']) + \
            sum(stat_dict['num_not_flipped_kept_attr'])
        # ==============================================================================================================
        logging.info('Evaluated {:d} seed samples'.format(num_all_seed_samples))
        logging.info('{:d} ({:.4f}%) TRUE source sentences have been translated CORRECTLY'
                     .format(stat_dict['num_true_samples_good_translations'],
                             _pc(stat_dict['num_true_samples_good_translations'], num_all_seed_samples)))

        logging.info('{:d} ({:.4f}%) TRUE source sentences have been translated INCORRECTLY'
                     .format(stat_dict['num_true_samples_bad_translations'],
                             _pc(stat_dict['num_true_samples_bad_translations'], num_all_seed_samples)))

        logging.info('{:d} ({:.4f}%) TRUE source sentences have been translated with a DELETED HOMOGRAPH'
                     .format(stat_dict['num_true_samples_deleted_homograph'],
                             _pc(stat_dict['num_true_samples_deleted_homograph'], num_all_seed_samples)))

        logging.info('{:d} ({:.4f}%) TRUE source sentences have been translated MAYBE INCORRECTLY'
                     .format(stat_dict['num_true_samples_maybe_bad_translations'],
                             _pc(stat_dict['num_true_samples_maybe_bad_translations'], num_all_seed_samples)))

        logging.info('{:d} ({:.4f}%) ALL TRUE source sentences have been translated WRONG'
                     .format(stat_dict['num_true_samples_bad_translations'] +
                             stat_dict['num_true_samples_deleted_homograph'] +
                             stat_dict['num_true_samples_maybe_bad_translations'],
                             100 - _pc(stat_dict['num_true_samples_good_translations'], num_all_seed_samples)))
        # ==============================================================================================================
        logging.info('-' * 20)
        num_all_adversarial_samples = sum(stat_dict['num_adversarial_samples'])
        logging.info('Evaluated {:d} unique adversarial samples'.format(num_all_adversarial_samples))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_adversarial_samples'][0],
                             _pc(stat_dict['num_adversarial_samples'][0], num_all_adversarial_samples)))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_adversarial_samples'][1],
                             _pc(stat_dict['num_adversarial_samples'][1], num_all_adversarial_samples)))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_adversarial_samples'][2],
                             _pc(stat_dict['num_adversarial_samples'][2], num_all_adversarial_samples)))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_adversarial_samples'][3],
                             _pc(stat_dict['num_adversarial_samples'][3], num_all_adversarial_samples)))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_adversarial_samples'][4],
                             _pc(stat_dict['num_adversarial_samples'][4], num_all_adversarial_samples)))
        # ==============================================================================================================
        logging.info('-' * 20)
        num_all_not_flipped = sum(stat_dict['num_not_flipped'])
        logging.info('{:d} ({:.4f}%) ADVERSARIAL source samples have been translated CORRECTLY'
                     .format(num_all_not_flipped,
                             _pc(num_all_not_flipped, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped'][0],
                             _pc(stat_dict['num_not_flipped'][0], stat_dict['num_adversarial_samples'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped'][1],
                             _pc(stat_dict['num_not_flipped'][1], stat_dict['num_adversarial_samples'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped'][2],
                             _pc(stat_dict['num_not_flipped'][2], stat_dict['num_adversarial_samples'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped'][3],
                             _pc(stat_dict['num_not_flipped'][3], stat_dict['num_adversarial_samples'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped'][4],
                             _pc(stat_dict['num_not_flipped'][4], stat_dict['num_adversarial_samples'][4])))

        logging.info('{:d} ({:.4f}%) ADVERSARIAL source samples have been translated INCORRECTLY'
                     .format(num_all_incorrect_adv_translations,
                             _pc(num_all_incorrect_adv_translations, num_all_adversarial_samples)))
        # ==============================================================================================================
        logging.info('-' * 20)
        num_all_flipped_to_attr = sum(stat_dict['num_flipped_to_attr_sense'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) INCORRECT adversarial translations were flipped to the ATTRACTOR\'S '
                     'target sense '
                     .format(num_all_flipped_to_attr,
                             _pc(num_all_flipped_to_attr, num_all_incorrect_adv_translations),
                             _pc(num_all_flipped_to_attr, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense'][0],
                             _pc(stat_dict['num_flipped_to_attr_sense'][0], stat_dict['num_adversarial_samples'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense'][1],
                             _pc(stat_dict['num_flipped_to_attr_sense'][1], stat_dict['num_adversarial_samples'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense'][2],
                             _pc(stat_dict['num_flipped_to_attr_sense'][2], stat_dict['num_adversarial_samples'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense'][3],
                             _pc(stat_dict['num_flipped_to_attr_sense'][3], stat_dict['num_adversarial_samples'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense'][4],
                             _pc(stat_dict['num_flipped_to_attr_sense'][4], stat_dict['num_adversarial_samples'][4])))

        num_all_flipped_to_other = sum(stat_dict['num_flipped_to_other_sense'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) INCORRECT adversarial translations were flipped to some OTHER KNOWN '
                     'target sense'
                     .format(num_all_flipped_to_other,
                             _pc(num_all_flipped_to_other, num_all_incorrect_adv_translations),
                             _pc(num_all_flipped_to_other, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense'][0],
                             _pc(stat_dict['num_flipped_to_other_sense'][0], stat_dict['num_adversarial_samples'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense'][1],
                             _pc(stat_dict['num_flipped_to_other_sense'][1], stat_dict['num_adversarial_samples'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense'][2],
                             _pc(stat_dict['num_flipped_to_other_sense'][2], stat_dict['num_adversarial_samples'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense'][3],
                             _pc(stat_dict['num_flipped_to_other_sense'][3], stat_dict['num_adversarial_samples'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense'][4],
                             _pc(stat_dict['num_flipped_to_other_sense'][4], stat_dict['num_adversarial_samples'][4])))

        num_all_deleted_homograph = sum(stat_dict['num_deleted_homograph'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) INCORRECT adversarial translations contained a DELETED homograph'
                     .format(num_all_deleted_homograph,
                             _pc(num_all_deleted_homograph, num_all_incorrect_adv_translations),
                             _pc(num_all_deleted_homograph, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph'][0],
                             _pc(stat_dict['num_deleted_homograph'][0], stat_dict['num_adversarial_samples'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph'][1],
                             _pc(stat_dict['num_deleted_homograph'][1], stat_dict['num_adversarial_samples'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph'][2],
                             _pc(stat_dict['num_deleted_homograph'][2], stat_dict['num_adversarial_samples'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph'][3],
                             _pc(stat_dict['num_deleted_homograph'][3], stat_dict['num_adversarial_samples'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph'][4],
                             _pc(stat_dict['num_deleted_homograph'][4], stat_dict['num_adversarial_samples'][4])))

        num_all_maybe_flipped = sum(stat_dict['num_maybe_flipped'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) INCORRECT adversarial translations were MAYBE flipped'
                     .format(num_all_maybe_flipped,
                             _pc(num_all_maybe_flipped, num_all_incorrect_adv_translations),
                             _pc(num_all_maybe_flipped, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped'][0],
                             _pc(stat_dict['num_maybe_flipped'][0], stat_dict['num_adversarial_samples'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped'][1],
                             _pc(stat_dict['num_maybe_flipped'][1], stat_dict['num_adversarial_samples'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped'][2],
                             _pc(stat_dict['num_maybe_flipped'][2], stat_dict['num_adversarial_samples'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped'][3],
                             _pc(stat_dict['num_maybe_flipped'][3], stat_dict['num_adversarial_samples'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped'][4],
                             _pc(stat_dict['num_maybe_flipped'][4], stat_dict['num_adversarial_samples'][4])))
        # ==============================================================================================================
        logging.info('=' * 20)
        logging.info('{:d} ({:.4f}%) adversarial translations DELETED the attractor term'
                     .format(num_all_deleted_attractor,
                             _pc(num_all_deleted_attractor, num_all_adversarial_samples)))

        logging.info('{:d} ({:.4f}%) adversarial translations KEPT the attractor term'
                     .format(num_all_kept_attractor,
                             _pc(num_all_kept_attractor, num_all_adversarial_samples)))
        # ==============================================================================================================
        logging.info('=' * 20)
        num_all_flipped_to_attr_del = sum(stat_dict['num_flipped_to_attr_sense_deleted_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations flipped to the ATTRACTOR\'S sense DELETED '
                     'the attractor'
                     .format(num_all_flipped_to_attr_del,
                             _pc(num_all_flipped_to_attr_del, num_all_flipped_to_attr),
                             _pc(num_all_flipped_to_attr_del, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_deleted_attr'][0],
                             _pc(stat_dict['num_flipped_to_attr_sense_deleted_attr'][0],
                                 stat_dict['num_flipped_to_attr_sense'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_deleted_attr'][1],
                             _pc(stat_dict['num_flipped_to_attr_sense_deleted_attr'][1],
                                 stat_dict['num_flipped_to_attr_sense'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_deleted_attr'][2],
                             _pc(stat_dict['num_flipped_to_attr_sense_deleted_attr'][2],
                                 stat_dict['num_flipped_to_attr_sense'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_deleted_attr'][3],
                             _pc(stat_dict['num_flipped_to_attr_sense_deleted_attr'][3],
                                 stat_dict['num_flipped_to_attr_sense'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_deleted_attr'][4],
                             _pc(stat_dict['num_flipped_to_attr_sense_deleted_attr'][4],
                                 stat_dict['num_flipped_to_attr_sense'][4])))

        num_all_flipped_to_other_del = sum(stat_dict['num_flipped_to_other_sense_deleted_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations flipped to OTHER senses DELETED the attractor'
                     .format(num_all_flipped_to_other_del,
                             _pc(num_all_flipped_to_other_del, num_all_flipped_to_other),
                             _pc(num_all_flipped_to_other_del, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_deleted_attr'][0],
                             _pc(stat_dict['num_flipped_to_other_sense_deleted_attr'][0],
                                 stat_dict['num_flipped_to_other_sense'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_deleted_attr'][1],
                             _pc(stat_dict['num_flipped_to_other_sense_deleted_attr'][1],
                                 stat_dict['num_flipped_to_other_sense'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_deleted_attr'][2],
                             _pc(stat_dict['num_flipped_to_other_sense_deleted_attr'][2],
                                 stat_dict['num_flipped_to_other_sense'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_deleted_attr'][3],
                             _pc(stat_dict['num_flipped_to_other_sense_deleted_attr'][3],
                                 stat_dict['num_flipped_to_other_sense'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_deleted_attr'][4],
                             _pc(stat_dict['num_flipped_to_other_sense_deleted_attr'][4],
                                 stat_dict['num_flipped_to_other_sense'][4])))

        num_all_deleted_homograph_del = sum(stat_dict['num_deleted_homograph_deleted_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations with DELETED homographs also DELETED the '
                     'attractor'
                     .format(num_all_deleted_homograph_del,
                             _pc(num_all_deleted_homograph_del, num_all_deleted_homograph),
                             _pc(num_all_deleted_homograph_del, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_deleted_attr'][0],
                             _pc(stat_dict['num_deleted_homograph_deleted_attr'][0],
                                 stat_dict['num_deleted_homograph'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_deleted_attr'][1],
                             _pc(stat_dict['num_deleted_homograph_deleted_attr'][1],
                                 stat_dict['num_deleted_homograph'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_deleted_attr'][2],
                             _pc(stat_dict['num_deleted_homograph_deleted_attr'][2],
                                 stat_dict['num_deleted_homograph'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_deleted_attr'][3],
                             _pc(stat_dict['num_deleted_homograph_deleted_attr'][3],
                                 stat_dict['num_deleted_homograph'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_deleted_attr'][4],
                             _pc(stat_dict['num_deleted_homograph_deleted_attr'][4],
                                 stat_dict['num_deleted_homograph'][4])))

        num_all_maybe_flipped_del = sum(stat_dict['num_maybe_flipped_deleted_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations that were MAYBE flipped DELETED the attractor'
                     .format(num_all_maybe_flipped_del,
                             _pc(num_all_maybe_flipped_del, num_all_maybe_flipped),
                             _pc(num_all_maybe_flipped_del, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_deleted_attr'][0],
                             _pc(stat_dict['num_maybe_flipped_deleted_attr'][0], stat_dict['num_maybe_flipped'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_deleted_attr'][1],
                             _pc(stat_dict['num_maybe_flipped_deleted_attr'][1], stat_dict['num_maybe_flipped'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_deleted_attr'][2],
                             _pc(stat_dict['num_maybe_flipped_deleted_attr'][2], stat_dict['num_maybe_flipped'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_deleted_attr'][3],
                             _pc(stat_dict['num_maybe_flipped_deleted_attr'][3], stat_dict['num_maybe_flipped'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_deleted_attr'][4],
                             _pc(stat_dict['num_maybe_flipped_deleted_attr'][4], stat_dict['num_maybe_flipped'][4])))

        num_all_not_flipped_del = sum(stat_dict['num_not_flipped_deleted_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations that were NOT flipped DELETED the attractor'
                     .format(num_all_not_flipped_del,
                             _pc(num_all_not_flipped_del, num_all_not_flipped),
                             _pc(num_all_not_flipped_del, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_deleted_attr'][0],
                             _pc(stat_dict['num_not_flipped_deleted_attr'][0], stat_dict['num_not_flipped'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_deleted_attr'][1],
                             _pc(stat_dict['num_not_flipped_deleted_attr'][1], stat_dict['num_not_flipped'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_deleted_attr'][2],
                             _pc(stat_dict['num_not_flipped_deleted_attr'][2], stat_dict['num_not_flipped'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_deleted_attr'][3],
                             _pc(stat_dict['num_not_flipped_deleted_attr'][3], stat_dict['num_not_flipped'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_deleted_attr'][4],
                             _pc(stat_dict['num_not_flipped_deleted_attr'][4], stat_dict['num_not_flipped'][4])))
        # ==============================================================================================================
        logging.info('=' * 20)
        num_all_flipped_to_attr_kept = sum(stat_dict['num_flipped_to_attr_sense_kept_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations flipped to the ATTRACTOR\'S sense KEPT '
                     'the attractor'
                     .format(num_all_flipped_to_attr_kept,
                             _pc(num_all_flipped_to_attr_kept, num_all_flipped_to_attr),
                             _pc(num_all_flipped_to_attr_kept,  num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_kept_attr'][0],
                             _pc(stat_dict['num_flipped_to_attr_sense_kept_attr'][0],
                                 stat_dict['num_flipped_to_attr_sense'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_kept_attr'][1],
                             _pc(stat_dict['num_flipped_to_attr_sense_kept_attr'][1],
                                 stat_dict['num_flipped_to_attr_sense'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_kept_attr'][2],
                             _pc(stat_dict['num_flipped_to_attr_sense_kept_attr'][2],
                                 stat_dict['num_flipped_to_attr_sense'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_kept_attr'][3],
                             _pc(stat_dict['num_flipped_to_attr_sense_kept_attr'][3],
                                 stat_dict['num_flipped_to_attr_sense'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_attr_sense_kept_attr'][4],
                             _pc(stat_dict['num_flipped_to_attr_sense_kept_attr'][4],
                                 stat_dict['num_flipped_to_attr_sense'][4])))

        num_all_flipped_to_other_kept = sum(stat_dict['num_flipped_to_other_sense_kept_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations flipped to OTHER senses KEPT the attractor'
                     .format(num_all_flipped_to_other_kept,
                             _pc(num_all_flipped_to_other_kept, num_all_flipped_to_other),
                             _pc(num_all_flipped_to_other_kept, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_kept_attr'][0],
                             _pc(stat_dict['num_flipped_to_other_sense_kept_attr'][0],
                                 stat_dict['num_flipped_to_other_sense'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_kept_attr'][1],
                             _pc(stat_dict['num_flipped_to_other_sense_kept_attr'][1],
                                 stat_dict['num_flipped_to_other_sense'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_kept_attr'][2],
                             _pc(stat_dict['num_flipped_to_other_sense_kept_attr'][2],
                                 stat_dict['num_flipped_to_other_sense'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_kept_attr'][3],
                             _pc(stat_dict['num_flipped_to_other_sense_kept_attr'][3],
                                 stat_dict['num_flipped_to_other_sense'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_flipped_to_other_sense_kept_attr'][4],
                             _pc(stat_dict['num_flipped_to_other_sense_kept_attr'][4],
                                 stat_dict['num_flipped_to_other_sense'][4])))

        num_all_deleted_homograph_kept = sum(stat_dict['num_deleted_homograph_kept_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations with DELETED homographs KEPT the '
                     'attractor'
                     .format(num_all_deleted_homograph_kept,
                             _pc(num_all_deleted_homograph_kept, num_all_deleted_homograph),
                             _pc(num_all_deleted_homograph_kept, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_kept_attr'][0],
                             _pc(stat_dict['num_deleted_homograph_kept_attr'][0],
                                 stat_dict['num_deleted_homograph'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_kept_attr'][1],
                             _pc(stat_dict['num_deleted_homograph_kept_attr'][1],
                                 stat_dict['num_deleted_homograph'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_kept_attr'][2],
                             _pc(stat_dict['num_deleted_homograph_kept_attr'][2],
                                 stat_dict['num_deleted_homograph'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_kept_attr'][3],
                             _pc(stat_dict['num_deleted_homograph_kept_attr'][3],
                                 stat_dict['num_deleted_homograph'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_deleted_homograph_kept_attr'][4],
                             _pc(stat_dict['num_deleted_homograph_kept_attr'][4],
                                 stat_dict['num_deleted_homograph'][4])))

        num_all_maybe_flipped_kept = sum(stat_dict['num_maybe_flipped_kept_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations that were MAYBE flipped KEPT the attractor'
                     .format(num_all_maybe_flipped_kept,
                             _pc(num_all_maybe_flipped_kept, num_all_maybe_flipped),
                             _pc(num_all_maybe_flipped_kept, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_kept_attr'][0],
                             _pc(stat_dict['num_maybe_flipped_kept_attr'][0], stat_dict['num_maybe_flipped'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_kept_attr'][1],
                             _pc(stat_dict['num_maybe_flipped_kept_attr'][1], stat_dict['num_maybe_flipped'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_kept_attr'][2],
                             _pc(stat_dict['num_maybe_flipped_kept_attr'][2], stat_dict['num_maybe_flipped'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_kept_attr'][3],
                             _pc(stat_dict['num_maybe_flipped_kept_attr'][3], stat_dict['num_maybe_flipped'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_maybe_flipped_kept_attr'][4],
                             _pc(stat_dict['num_maybe_flipped_kept_attr'][4], stat_dict['num_maybe_flipped'][4])))

        num_all_not_flipped_kept = sum(stat_dict['num_not_flipped_kept_attr'])
        logging.info('{:d} ({:.4f}%) / ({:.4f}%) adversarial translations that were NOT flipped KEPT the attractor'
                     .format(num_all_not_flipped_kept,
                             _pc(num_all_not_flipped_kept, num_all_not_flipped),
                             _pc(num_all_not_flipped_kept, num_all_adversarial_samples)))
        logging.info('Correct seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_kept_attr'][0],
                             _pc(stat_dict['num_not_flipped_kept_attr'][0], stat_dict['num_not_flipped'][0])))
        logging.info('Incorrect seed translation (attractor sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_kept_attr'][1],
                             _pc(stat_dict['num_not_flipped_kept_attr'][1], stat_dict['num_not_flipped'][1])))
        logging.info('Incorrect seed translation (other sense): {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_kept_attr'][2],
                             _pc(stat_dict['num_not_flipped_kept_attr'][2], stat_dict['num_not_flipped'][2])))
        logging.info('Deleted homograph seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_kept_attr'][3],
                             _pc(stat_dict['num_not_flipped_kept_attr'][3], stat_dict['num_not_flipped'][3])))
        logging.info('Maybe incorrect seed translation: {:d} ({:.4f}%)'
                     .format(stat_dict['num_not_flipped_kept_attr'][4],
                             _pc(stat_dict['num_not_flipped_kept_attr'][4], stat_dict['num_not_flipped'][4])))
        # ==============================================================================================================
        logging.info('=' * 20)
        logging.info('Number of samples per generation strategy:')
        num_all_samples = sum(stat_dict['all_samples_method_count'])
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['all_samples_method_count'][method_id],
                                 _pc(stat_dict['all_samples_method_count'][method_id], num_all_samples)))
        logging.info('Number of to-attractor-sense flips per generation strategy (correct seed translations):')
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['flipped_to_attr_sense_method_count_good_seed'][method_id],
                                 _pc(stat_dict['flipped_to_attr_sense_method_count_good_seed'][method_id],
                                     stat_dict['good_seed_translations_method_count'][method_id])))
        logging.info('Number of to-attractor-sense flips per generation strategy (other sense seed translations):')
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['flipped_to_attr_sense_method_count_other_seed'][method_id],
                                 _pc(stat_dict['flipped_to_attr_sense_method_count_other_seed'][method_id],
                                     stat_dict['other_sense_seed_translations_method_count'][method_id])))
        logging.info('Number of homograph deletions per generation strategy (correct seed translations):')
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['deleted_homograph_method_count_good_seed'][method_id],
                                 _pc(stat_dict['deleted_homograph_method_count_good_seed'][method_id],
                                     stat_dict['good_seed_translations_method_count'][method_id])))
        logging.info('Number of homograph deletions per generation strategy (attractor sense seed translations):')
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['deleted_homograph_method_count_attr_seed'][method_id],
                                 _pc(stat_dict['deleted_homograph_method_count_attr_seed'][method_id],
                                     stat_dict['attr_sense_seed_translations_method_count'][method_id])))
        logging.info('Number of homograph deletions per generation strategy (other sense seed translations):')
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['deleted_homograph_method_count_other_seed'][method_id],
                                 _pc(stat_dict['deleted_homograph_method_count_other_seed'][method_id],
                                     stat_dict['other_sense_seed_translations_method_count'][method_id])))
        logging.info('Number of attractor deletions per generation strategy:')
        for method_id, gen_method in enumerate(GEN_METHODS):
            logging.info('{:s}: {:d} ({:.4f}%)'
                         .format(gen_method, stat_dict['deleted_attr_method_count'][method_id],
                                 _pc(stat_dict['deleted_attr_method_count'][method_id],
                                     stat_dict['all_samples_method_count'][method_id])))

    # Read-in adversarial samples
    logging.info('Reading-in adversarial samples table ...')
    with open(adversarial_samples_path, 'r', encoding='utf8') as asp:
        adversarial_samples_table = json.load(asp)

    # Read-in attractor table
    logging.info('Reading-in attractor table ...')
    with open(attractors_path, 'r', encoding='utf8') as ap:
        attractors_table = json.load(ap)

    # Load sense to cluster mappings
    logging.info('Reading-in sense clusters ...')
    with open(sense_clusters_path, 'r', encoding='utf8') as scp:
        sense_clusters_table = json.load(scp)

    # Add blacklisted, ambiguous terms to the set of acceptable translations
    for term in sense_clusters_table.keys():
        for cluster in sense_clusters_table[term].keys():
            if '[AMBIGUOUS SENSES]' in sense_clusters_table[term][cluster].keys():
                sense_clusters_table[term][cluster]['[SENSES]'] += \
                    sense_clusters_table[term][cluster]['[AMBIGUOUS SENSES]']
                sense_clusters_table[term][cluster].pop('[AMBIGUOUS SENSES]')
            if '[BLACKLISTED SENSES]' in sense_clusters_table[term][cluster].keys():
                sense_clusters_table[term][cluster]['[SENSES]'] += \
                    sense_clusters_table[term][cluster]['[BLACKLISTED SENSES]']
                sense_clusters_table[term][cluster].pop('[BLACKLISTED SENSES]')
            sense_clusters_table[term][cluster]['[SENSES]'] = \
                list(set(sense_clusters_table[term][cluster]['[SENSES]']))
    sense_to_cluster_table = _build_cluster_lookup(sense_clusters_table)

    # Post-process sense-to-cluster table
    sense_lemmas_to_cluster_table = sense_to_cluster_table
    sense_tokens_to_cluster_table = dict()
    for src_term in sense_lemmas_to_cluster_table.keys():
        sense_tokens_to_cluster_table[src_term] = dict()
        for cluster_tuple_list in sense_lemmas_to_cluster_table[src_term].values():
            for cluster_tuple in cluster_tuple_list:
                sense_tokens_to_cluster_table[src_term][cluster_tuple[0].lower()] = [cluster_tuple]

    cluster_to_sense_lemmas_table = dict()
    # Derive a cluster-to-senses table, used for compound analysis
    for src_term in sense_lemmas_to_cluster_table.keys():
        cluster_to_sense_lemmas_table[src_term] = dict()
        for sense, clusters in sense_lemmas_to_cluster_table[src_term].items():
            for cls_tpl in clusters:
                if not cluster_to_sense_lemmas_table[src_term].get(cls_tpl[1], None):
                    cluster_to_sense_lemmas_table[src_term][cls_tpl[1]] = [sense]
                else:
                    if sense not in cluster_to_sense_lemmas_table[src_term][cls_tpl[1]]:
                        cluster_to_sense_lemmas_table[src_term][cls_tpl[1]].append(sense)

    cluster_to_sense_tokens_table = dict()
    # Derive a cluster-to-senses table, used for compound analysis
    for src_term in sense_tokens_to_cluster_table.keys():
        cluster_to_sense_tokens_table[src_term] = dict()
        for sense, clusters in sense_lemmas_to_cluster_table[src_term].items():
            for cls_tpl in clusters:
                if not cluster_to_sense_tokens_table[src_term].get(cls_tpl[1], None):
                    cluster_to_sense_tokens_table[src_term][cls_tpl[1]] = [cls_tpl[0]]
                else:
                    if sense not in cluster_to_sense_tokens_table[src_term][cls_tpl[1]]:
                        cluster_to_sense_tokens_table[src_term][cls_tpl[1]].append(cls_tpl[0])

    # Restructure NMT translations for easier access
    logging.info('-' * 10)
    logging.info('Hashing true challenge set samples ...')
    with open(true_sources_path, 'r', encoding='utf8') as tsp:
        true_sources = [line for line_id, line in enumerate(tsp)]
    with open(true_translations_path, 'r', encoding='utf8') as ttp:
        true_translations = [line for line_id, line in enumerate(ttp)]
    with open(true_alignments_path, 'r', encoding='utf8') as tap:
        true_alignments = [line for line_id, line in enumerate(tap)]

    logging.info('Hashing adversarial set samples ...')
    with open(adversarial_sources_path, 'r', encoding='utf8') as asp:
        adv_sources = [line for line_id, line in enumerate(asp)]
    with open(adversarial_translations_path, 'r', encoding='utf8') as atp:
        adv_translations = [line for line_id, line in enumerate(atp)]
    with open(adversarial_alignments_path, 'r', encoding='utf8') as aap:
        adversarial_alignments = [line for line_id, line in enumerate(aap)]
    logging.info('-' * 10)

    # Create a adversarial-to-original map
    adv_to_sample = dict()
    for term in adversarial_samples_table.keys():
        for true_cluster_id in adversarial_samples_table[term].keys():
            for adv_cluster_id in adversarial_samples_table[term][true_cluster_id].keys():
                for smp in adversarial_samples_table[term][true_cluster_id][adv_cluster_id]:
                    adv = smp[0].strip()
                    adv_to_sample[adv] = (smp, term)

    # Combine sources, samples, translations, and alignments
    logging.info('Mapping translations...')
    adv_to_info = {adv_sources_line: [true_sources[line_id],
                                      true_translations[line_id],
                                      adv_translations[line_id],
                                      true_alignments[line_id],
                                      adversarial_alignments[line_id],
                                      adv_to_sample[adv_sources_line.strip()][0],
                                      adv_to_sample[adv_sources_line.strip()][1]] for line_id, adv_sources_line
                   in enumerate(adv_sources)}

    # For stats
    unique_samples = dict()

    # Seed translations
    true_samples_good_translations = dict()
    true_samples_bad_translations = dict()
    true_samples_deleted_homograph = dict()
    true_samples_maybe_bad_translations = dict()

    # Attack success tables
    flipped_to_attr_sense_adv_samples = dict()
    flipped_to_other_sense_adv_samples = dict()
    deleted_homograph_adv_samples = dict()
    maybe_flipped_adv_samples = dict()
    not_flipped_adv_samples = dict()

    # Attractor deletion tables
    flipped_to_attr_sense_adv_samples_deleted_attr = dict()
    flipped_to_other_sense_adv_samples_deleted_attr = dict()
    deleted_homograph_adv_samples_deleted_attr = dict()
    maybe_flipped_adv_samples_deleted_attr = dict()
    not_flipped_adv_samples_deleted_attr = dict()
    all_adv_samples_deleted_attractor = dict()

    flipped_to_attr_sense_adv_samples_kept_attr = dict()
    flipped_to_other_sense_adv_samples_kept_attr = dict()
    deleted_homograph_adv_samples_kept_attr = dict()
    maybe_flipped_adv_samples_kept_attr = dict()
    not_flipped_adv_samples_kept_attr = dict()
    all_adv_samples_kept_attractor = dict()

    # Initialize variables for reporting
    stat_dict = {
        'num_true_samples_good_translations': 0,
        'num_true_samples_bad_translations': 0,
        'num_true_samples_deleted_homograph': 0,
        'num_true_samples_maybe_bad_translations': 0,

        'num_adversarial_samples': [0, 0, 0, 0, 0],
        'num_flipped_to_attr_sense': [0, 0, 0, 0, 0],
        'num_flipped_to_other_sense': [0, 0, 0, 0, 0],
        'num_deleted_homograph': [0, 0, 0, 0, 0],
        'num_maybe_flipped': [0, 0, 0, 0, 0],
        'num_not_flipped': [0, 0, 0, 0, 0],

        'num_flipped_to_attr_sense_deleted_attr': [0, 0, 0, 0, 0],
        'num_flipped_to_other_sense_deleted_attr': [0, 0, 0, 0, 0],
        'num_deleted_homograph_deleted_attr': [0, 0, 0, 0, 0],
        'num_maybe_flipped_deleted_attr': [0, 0, 0, 0, 0],
        'num_not_flipped_deleted_attr': [0, 0, 0, 0, 0],

        'num_flipped_to_attr_sense_kept_attr': [0, 0, 0, 0, 0],
        'num_flipped_to_other_sense_kept_attr': [0, 0, 0, 0, 0],
        'num_deleted_homograph_kept_attr': [0, 0, 0, 0, 0],
        'num_maybe_flipped_kept_attr': [0, 0, 0, 0, 0],
        'num_not_flipped_kept_attr': [0, 0, 0, 0, 0],

        'all_samples_method_count': [0, 0, 0, 0],
        'good_seed_translations_method_count': [0, 0, 0, 0],
        'attr_sense_seed_translations_method_count': [0, 0, 0, 0],
        'other_sense_seed_translations_method_count': [0, 0, 0, 0],
        'flipped_to_attr_sense_method_count_good_seed': [0, 0, 0, 0],
        'flipped_to_attr_sense_method_count_other_seed': [0, 0, 0, 0],
        'deleted_homograph_method_count_good_seed': [0, 0, 0, 0],
        'deleted_homograph_method_count_attr_seed': [0, 0, 0, 0],
        'deleted_homograph_method_count_other_seed': [0, 0, 0, 0],
        'deleted_attr_method_count': [0, 0, 0, 0]
    }

    for adv_sent_id, adv_sent in enumerate(adv_to_info.keys()):

        sample = adv_to_info[adv_sent]
        term = sample[6]

        # TODO: DEBUGGING
        # if term not in ['anchor', 'clip']:
        #     continue

        # Apply adversarial filtering and compute LM-based fluency / acceptability scores
        _score_and_filter(sample, attractors_table[term], term)

        # Occasionally report statistics
        if adv_sent_id > 0 and adv_sent_id % 1000 == 0:
            logging.info('Looked up {:d} terms; reporting intermediate statistics:'.format(adv_sent_id))
            _show_stats()

        # TODO: DEBUGGING
        # if term_id == 2:
        #     break

    # Final report
    logging.info('Looked up {:d} terms; reporting FINAL statistics:'.format(len(adversarial_samples_table.keys())))
    _show_stats()

    # Construct output paths
    table_list = [
        true_samples_good_translations,
        true_samples_bad_translations,
        true_samples_deleted_homograph,
        true_samples_maybe_bad_translations,

        flipped_to_attr_sense_adv_samples,
        flipped_to_other_sense_adv_samples,
        deleted_homograph_adv_samples,
        maybe_flipped_adv_samples,
        not_flipped_adv_samples,

        flipped_to_attr_sense_adv_samples_deleted_attr,
        flipped_to_other_sense_adv_samples_deleted_attr,
        deleted_homograph_adv_samples_deleted_attr,
        maybe_flipped_adv_samples_deleted_attr,
        not_flipped_adv_samples_deleted_attr,
        all_adv_samples_deleted_attractor,

        flipped_to_attr_sense_adv_samples_kept_attr,
        flipped_to_other_sense_adv_samples_kept_attr,
        deleted_homograph_adv_samples_kept_attr,
        maybe_flipped_adv_samples_kept_attr,
        not_flipped_adv_samples_kept_attr,
        all_adv_samples_kept_attractor
    ]

    attack_success_path = os.path.join(output_tables_dir, 'attack_success')

    deleted_attractors_path = os.path.join(output_tables_dir, 'deleted_attractors')
    kept_attractors_path = os.path.join(output_tables_dir, 'kept_attractors')

    for out_dir in [attack_success_path,
                    deleted_attractors_path, kept_attractors_path]:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    path_list = [
        os.path.join(attack_success_path, 'true_samples_good_translations.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'true_samples_bad_translations.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'true_samples_deleted_homograph.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'true_samples_maybe_bad_translations.{:s}'.format(src_lang)),

        os.path.join(attack_success_path, 'flipped_to_attr_sense_adv_samples.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'flipped_to_other_sense_adv_samples.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'deleted_homograph.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'maybe_flipped_adv_samples.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'not_flipped_adv_samples.{:s}'.format(src_lang)),

        os.path.join(deleted_attractors_path, 'flipped_to_attr_sense_adv_samples_deleted_attr.{:s}'.format(src_lang)),
        os.path.join(deleted_attractors_path, 'flipped_to_other_sense_adv_samples_deleted_attr.{:s}'.format(src_lang)),
        os.path.join(deleted_attractors_path, 'deleted_homograph_adv_samples_deleted_attr.{:s}'.format(src_lang)),
        os.path.join(deleted_attractors_path, 'maybe_flipped_adv_samples_deleted_attr.{:s}'.format(src_lang)),
        os.path.join(deleted_attractors_path, 'not_flipped_adv_samples_deleted_attr.{:s}'.format(src_lang)),
        os.path.join(deleted_attractors_path, 'all_adv_samples_deleted_attr.{:s}'.format(src_lang)),

        os.path.join(kept_attractors_path, 'flipped_to_attr_sense_adv_samples_kept_attr.{:s}'.format(src_lang)),
        os.path.join(kept_attractors_path, 'flipped_to_other_sense_adv_samples_kept_attr.{:s}'.format(src_lang)),
        os.path.join(kept_attractors_path, 'deleted_homograph_adv_samples_kept_attr.{:s}'.format(src_lang)),
        os.path.join(kept_attractors_path, 'maybe_flipped_adv_samples_kept_attr.{:s}'.format(src_lang)),
        os.path.join(kept_attractors_path, 'not_flipped_adv_samples_kept_attr.{:s}'.format(src_lang)),
        os.path.join(kept_attractors_path, 'all_adv_samples_kept_attr.{:s}'.format(src_lang))
    ]

    # Save
    for table, path in zip(table_list, path_list):
        with open(path + '.json', 'w', encoding='utf8') as json_file:
            json.dump(table, json_file, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_samples_path', type=str, required=True,
                        help='path to the JSON file containing adversarial translation samples generated via the '
                             'introduction of semantic attractors')
    parser.add_argument('--true_sources_path', type=str, default=None,
                        help='path to the text file containing the true source sentences')
    parser.add_argument('--adversarial_sources_path', type=str, default=None,
                        help='path to the text file containing the perturbed source sentences')
    parser.add_argument('--true_translations_path', type=str, default=None,
                        help='path to the translations produced by the filter NMT model for the true source sentences')
    parser.add_argument('--adversarial_translations_path', type=str, default=None,
                        help='path to the translations produced by the filter NMT model for the adversarially '
                             'perturbed source sentences')
    parser.add_argument('--true_alignments_path', type=str, default=None,
                        help='path to file containing the fastalign alignments for the translations of the original '
                             'challenge sentences')
    parser.add_argument('--adversarial_alignments_path', type=str, default=None,
                        help='path to file containing the fastalign alignments for the translations of the '
                             'adversarially perturbed challenge sentences')
    parser.add_argument('--attractors_path', type=str, required=True,
                        help='path to the JSON file containing the extracted attractor terms')
    parser.add_argument('--sense_clusters_path', type=str, default=None,
                        help='path to the JSON file containing scraped BabelNet sense clusters')
    parser.add_argument('--output_tables_dir', type=str, required=True,
                        help='path to the destination of the filtered adversarial samples tables')
    parser.add_argument('--lang_pair', type=str, default=None,
                        help='language pair of the bitext; expected format is src-tgt')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.adversarial_samples_path.split('/')[:-1])
    file_name = args.adversarial_samples_path.split('/')[-1]
    file_name = '.'.join(file_name.split('.')[:-1])
    file_name = file_name if len(file_name) > 0 else 'log'
    log_dir = '{:s}/logs/'.format(base_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = '{:s}{:s}.log'.format(log_dir, file_name)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(levelname)s: %(message)s')
    # Logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # Instantiate processing pipeline
    src_lang, tgt_lang = args.lang_pair.strip().split('-')
    spacy_map = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
    try:
        src_nlp = spacy.load(spacy_map[src_lang], disable=['parser', 'ner', 'textcat'])
        tgt_nlp = spacy.load(spacy_map[tgt_lang], disable=['parser', 'ner', 'textcat'])
    except KeyError:
        logging.info('SpaCy does not support the language {:s} or {:s}. Exiting.'.format(src_lang, tgt_lang))
        sys.exit(0)
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    if src_lang == 'en':
        STOP_WORDS = stopwords.words('english')
    else:
        STOP_WORDS = []

    evaluate_attack_success(args.adversarial_samples_path,
                            args.true_sources_path,
                            args.adversarial_sources_path,
                            args.true_translations_path,
                            args.adversarial_translations_path,
                            args.true_alignments_path,
                            args.adversarial_alignments_path,
                            args.attractors_path,
                            args.sense_clusters_path,
                            args.output_tables_dir)
