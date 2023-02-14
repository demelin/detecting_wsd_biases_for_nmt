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
                   src,
                   ambiguous_token,
                   ambiguous_token_loc,
                   seed_cluster_id,
                   adv_cluster_id,
                   other_cluster_ids,
                   sense_lemmas_to_cluster,
                   cluster_to_sense_lemmas,
                   sense_tokens_to_cluster,
                   cluster_to_sense_tokens,
                   alignments,
                   tgt_sent):

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
    src_tokens = src.strip().split()
    ambiguous_token_loc = ambiguous_token_loc[0]

    # Check that the provided locations of the ambiguous token are correct
    ambiguous_token_in_src = src_tokens[ambiguous_token_loc].lower().strip(punctuation_plus_space)
    assert ambiguous_token.lower().strip(punctuation_plus_space) == \
        ambiguous_token_in_src or ambiguous_token[:-1] in ambiguous_token_in_src, \
        'Ambiguous token \'{:s}\' does not match the true source token \'{:s}\' at the token location'\
        .format(ambiguous_token, ambiguous_token_in_src)

    other_cluster_lemmas = list()
    for cluster_id in other_cluster_ids:
        other_cluster_lemmas += cluster_to_sense_lemmas[cluster_id]
    other_cluster_tokens = list()
    for cluster_id in other_cluster_ids:
        other_cluster_tokens += cluster_to_sense_tokens[cluster_id]

    target_hits = list()
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
    if homograph_aligned:
        homograph_translated = True
    else:
        if len(target_hits) > 0:
            homograph_translated = True
        else:
            homograph_translated = False

    # TODO / NOTE: Fixing a corpus bug
    if 'rendezvous' in cluster_to_sense_lemmas[seed_cluster_id]:
        temp = seed_cluster_id
        seed_cluster_id = adv_cluster_id
        adv_cluster_id = temp

    # Flatten target hits
    target_hits = [hit[1] for hit_list in target_hits for hit in hit_list]
    # If target term is ambiguous, assume the translation is correct
    if seed_cluster_id in target_hits:

        return 'not_flipped', target_hits

    elif adv_cluster_id in target_hits:

        # TODO: DEBUGGING
        logging.info('-' * 10)
        logging.info('FLIPPED TO ATTR')
        logging.info(src)
        logging.info(translation)
        logging.info(tgt_sent)
        logging.info('SEED SENSES')
        logging.info(cluster_to_sense_lemmas[seed_cluster_id])
        logging.info('ADVERSARIAL SENSES')
        logging.info(cluster_to_sense_lemmas[adv_cluster_id])

        return 'flipped_to_attr', target_hits

    elif len(set(other_cluster_ids) & set(target_hits)) >= 1:
        return 'flipped_to_other', target_hits

    # i.e. target_hits is empty
    else:
        if homograph_translated:
            return 'maybe_flipped', target_hits

        else:
            return 'deleted_homograph', target_hits


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


def evaluate_attack_success(wsd_samples_path,
                            sources_path,
                            translations_path,
                            alignments_path,
                            attractors_path,
                            sense_clusters_path,
                            output_tables_dir):

    """ Detects successful attacks and computes correlation between attack success and various metrics. """

    def _score_and_filter(combined_sample, ambiguous_term, attractors_entry):
        """ Helper function for filtering a single adversarial sample. """

        # Combined sample contents:

        # [translations[line_id],
        # alignments[line_id],
        # src_to_sample[src_line.strip()][0],
        # src_to_sample[src_line.strip()][1]]

        sense_lemmas_to_cluster = sense_lemmas_to_cluster_table.get(ambiguous_term, None)
        sense_tokens_to_cluster = sense_tokens_to_cluster_table.get(ambiguous_term, None)
        cluster_to_sense_lemmas = cluster_to_sense_lemmas_table.get(ambiguous_term, None)
        cluster_to_sense_tokens = cluster_to_sense_tokens_table.get(ambiguous_term, None)

        # Unpack
        translation = combined_sample[0]
        line_alignments = combined_sample[1]
        challenge_entry = combined_sample[2]

        src_sent = challenge_entry[1]
        tgt_sent = challenge_entry[2]
        ambiguous_token_loc_seed = challenge_entry[7]
        ambiguous_token_ws_loc_seed = challenge_entry[8]  # list
        seed_cluster_num = challenge_entry[12]
        adv_cluster_num = challenge_entry[13]

        other_cluster_ids = list(attractors_entry.keys())
        other_cluster_ids.pop(other_cluster_ids.index(seed_cluster_num))
        try:
            other_cluster_ids.pop(other_cluster_ids.index(adv_cluster_num))
        except ValueError:
            pass

        # Ignore samples containing multiple instances of the attractor term
        src_tokens = [tok.strip(punctuation_plus_space) for tok in src_sent.split()]
        src_tokens = [tok for tok in src_tokens if len(tok) > 0]
        # Ignore short sentences
        if len(src_tokens) < 10:
            return None

        # Get NMT labels
        true_nmt_label, true_translation_sense_clusters = \
            _get_nmt_label(translation,
                           src_sent,
                           ambiguous_term,
                           ambiguous_token_ws_loc_seed,
                           seed_cluster_num,
                           adv_cluster_num,
                           other_cluster_ids,
                           sense_lemmas_to_cluster,
                           cluster_to_sense_lemmas,
                           sense_tokens_to_cluster,
                           cluster_to_sense_tokens,
                           line_alignments,
                           tgt_sent)

        # Sort true samples into appropriate output tables, based on filtering outcome
        seed_table_to_expand = None
        if true_nmt_label == 'not_flipped':
            if not unique_samples.get((src_sent, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_good_translations'] += 1
                unique_samples[(src_sent, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_good_translations
        elif true_nmt_label == 'flipped_to_attr':
            if not unique_samples.get((src_sent, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_bad_translations'] += 1
                unique_samples[(src_sent, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_bad_translations
        elif true_nmt_label == 'flipped_to_other':
            if not unique_samples.get((src_sent, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_bad_translations'] += 1
                unique_samples[(src_sent, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_bad_translations
        elif true_nmt_label == 'deleted_homograph':
            if not unique_samples.get((src_sent, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_deleted_homograph'] += 1
                unique_samples[(src_sent, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_deleted_homograph
        else:
            if not unique_samples.get((src_sent, ambiguous_term, ambiguous_token_loc_seed), None):
                stat_dict['num_true_samples_maybe_bad_translations'] += 1
                unique_samples[(src_sent, ambiguous_term, ambiguous_token_loc_seed)] = True
                seed_table_to_expand = true_samples_maybe_bad_translations

        # Collect seed translations
        if seed_table_to_expand is not None:
            if not seed_table_to_expand.get(ambiguous_term, None):
                seed_table_to_expand[ambiguous_term] = dict()
            if not seed_table_to_expand[ambiguous_term].get(seed_cluster_num, None):
                seed_table_to_expand[ambiguous_term][seed_cluster_num] = dict()
            if not seed_table_to_expand[ambiguous_term][seed_cluster_num].get(adv_cluster_num, None):
                seed_table_to_expand[ambiguous_term][seed_cluster_num][adv_cluster_num] = dict()
            if not seed_table_to_expand[ambiguous_term][seed_cluster_num][adv_cluster_num].get(src_sent, None):
                seed_table_to_expand[ambiguous_term][seed_cluster_num][adv_cluster_num][src_sent] = list()
            seed_table_to_expand[ambiguous_term][seed_cluster_num][adv_cluster_num][src_sent]\
                .append([translation, tgt_sent, true_translation_sense_clusters, ambiguous_token_loc_seed])

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

    # Read-in adversarial samples
    logging.info('Reading-in challenge samples table ...')
    with open(wsd_samples_path, 'r', encoding='utf8') as wsp:
        wsd_samples_table = json.load(wsp)

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
    logging.info('Hashing challenge set samples ...')
    with open(sources_path, 'r', encoding='utf8') as sp:
        sources = [line for line_id, line in enumerate(sp)]
    with open(translations_path, 'r', encoding='utf8') as tp:
        translations = [line for line_id, line in enumerate(tp)]
    with open(alignments_path, 'r', encoding='utf8') as ap:
        alignments = [line for line_id, line in enumerate(ap)]
    logging.info('-' * 10)

    # Create a adversarial-to-original map
    src_to_sample = dict()
    for term in wsd_samples_table.keys():
        for true_cluster_id in wsd_samples_table[term].keys():
            for adv_cluster_id in wsd_samples_table[term][true_cluster_id].keys():
                for smp in wsd_samples_table[term][true_cluster_id][adv_cluster_id]:
                    src = smp[1].strip()
                    src_to_sample[src] = (smp, term)

    # Combine sources, samples, translations, and alignments
    logging.info('Mapping translations...')
    src_to_info = {src_line: [translations[line_id],
                              alignments[line_id],
                              src_to_sample[src_line.strip()][0],
                              src_to_sample[src_line.strip()][1]] for line_id, src_line
                   in enumerate(sources)}

    # For stats
    unique_samples = dict()

    # Seed translations
    true_samples_good_translations = dict()
    true_samples_bad_translations = dict()
    true_samples_deleted_homograph = dict()
    true_samples_maybe_bad_translations = dict()

    # Initialize variables for reporting
    stat_dict = {
        'num_true_samples_good_translations': 0,
        'num_true_samples_bad_translations': 0,
        'num_true_samples_deleted_homograph': 0,
        'num_true_samples_maybe_bad_translations': 0,
    }

    sent_id = 0
    for sent_id, sent in enumerate(src_to_info.keys()):

        sample = src_to_info[sent]
        term = sample[3]


        # TODO: DEBUGGING
        # if term not in ['anchor', 'clip']:
        #     continue

        # Apply adversarial filtering and compute LM-based fluency / acceptability scores
        _score_and_filter(sample, term, attractors_table[term])

        # Occasionally report statistics
        if sent_id > 0 and sent_id % 1000 == 0:
            logging.info('\nLooked up {:d} sentences; reporting intermediate statistics:'.format(sent_id + 1))
            _show_stats()

        # TODO: DEBUGGING
        # if term_id == 2:
        #     break

    # Final report
    logging.info('\nLooked up {:d} sentences; reporting FINAL statistics:'.format(sent_id + 1))
    _show_stats()

    # Construct output paths
    table_list = [
        true_samples_good_translations,
        true_samples_bad_translations,
        true_samples_deleted_homograph,
        true_samples_maybe_bad_translations,
    ]

    attack_success_path = os.path.join(output_tables_dir, 'attack_success')

    if not os.path.isdir(attack_success_path):
        os.mkdir(attack_success_path)

    path_list = [
        os.path.join(attack_success_path, 'wsd_samples_good_translations.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'wsd_samples_bad_translations.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'wsd_samples_deleted_homograph.{:s}'.format(src_lang)),
        os.path.join(attack_success_path, 'wsd_samples_maybe_bad_translations.{:s}'.format(src_lang)),
    ]

    # Save
    for table, path in zip(table_list, path_list):
        with open(path + '.json', 'w', encoding='utf8') as json_file:
            json.dump(table, json_file, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsd_samples_path', type=str, required=True,
                        help='path to the JSON file containing selected challenge samples')
    parser.add_argument('--sources_path', type=str, default=None,
                        help='path to the text file containing the true source sentences')
    parser.add_argument('--translations_path', type=str, default=None,
                        help='path to the translations produced by the filter NMT model for the true source sentences')
    parser.add_argument('--alignments_path', type=str, default=None,
                        help='path to file containing the fastalign alignments for the translations of the original '
                             'challenge sentences')
    parser.add_argument('--sense_clusters_path', type=str, default=None,
                        help='path to the JSON file containing scraped BabelNet sense clusters')
    parser.add_argument('--attractors_path', type=str, required=True,
                        help='path to the JSON file containing the extracted attractor terms')
    parser.add_argument('--output_tables_dir', type=str, required=True,
                        help='path to the destination of the filtered adversarial samples tables')
    parser.add_argument('--lang_pair', type=str, default=None,
                        help='language pair of the bitext; expected format is src-tgt')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.wsd_samples_path.split('/')[:-1])
    file_name = args.wsd_samples_path.split('/')[-1]
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

    evaluate_attack_success(args.wsd_samples_path,
                            args.sources_path,
                            args.translations_path,
                            args.alignments_path,
                            args.attractors_path,
                            args.sense_clusters_path,
                            args.output_tables_dir)
