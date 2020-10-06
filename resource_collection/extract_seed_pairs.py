""" Script for extracting source sentences containing the ambiguous terms from the specified corpora to be used in the
generation of adversarial samples; in essence a modified variant of extract_attractor_terms_bad_alignments.py """

import os
import re
import sys
import json
import spacy
import string
import logging
import argparse

import numpy as np
from string import punctuation
from nltk.corpus import stopwords

POLYSEMOUS_NOUNS = ['anchor',
                    'arm',
                    'band',
                    'bank',
                    'balance',
                    'bar',
                    'barrel',
                    'bark',
                    'bass',
                    'bat',
                    'battery',
                    'beam',
                    'board',
                    'bolt',
                    'boot',
                    'bow',
                    'brace',
                    'break',
                    'bug',
                    'butt',
                    'cabinet',
                    'capital',
                    'case',
                    'cast',
                    'chair',
                    'change',
                    'charge',
                    'chest',
                    'chip',
                    'clip',
                    'club',
                    'cock',
                    'counter',
                    'crane',
                    'cycle',
                    'date',
                    'deck',
                    'drill',
                    'drop',
                    'fall',
                    'fan',
                    'file',
                    'film',
                    'flat',
                    'fly',
                    'gum',
                    'hoe',
                    'hood',
                    'jam',
                    'jumper',
                    'lap',
                    'lead',
                    'letter',
                    'lock',
                    'mail',
                    'match',
                    'mine',
                    'mint',
                    'mold',
                    'mole',
                    'mortar',
                    'move',
                    'nail',
                    'note',
                    'offense',
                    'organ',
                    'pack',
                    'palm',
                    'pick',
                    'pitch',
                    'pitcher',
                    'plaster',
                    'plate',
                    'plot',
                    'pot',
                    'present',
                    'punch',
                    'quarter',
                    'race',
                    'racket',
                    'record',
                    'ruler',
                    'seal',
                    'sewer',
                    'scale',
                    'snare',
                    'spirit',
                    'spot',
                    'spring',
                    'staff',
                    'stock',
                    'subject',
                    'tank',
                    'tear',
                    'term',
                    'tie',
                    'toast',
                    'trunk',
                    'tube',
                    'vacuum',
                    'watch']

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


def collect_seed_samples(source_file_path,
                         target_file_path,
                         alignments_file_path,
                         sense_clusters_path,
                         attractors_path,
                         out_file_path,
                         filter_by_pos,
                         remove_stopwords,
                         verbose,
                         merge_source_forms=True):

    """ Collects parallel sentence pairs containing ambiguous source terms from the specified bi-text and assigns
    them to one of the known target sense clusters. """

    # Check file type
    is_nouns_only = '_nouns' in sense_clusters_path

    # Load-in pre-extracted BabelNet target sense clusters
    logging.info('Preparing target sense tables ...')
    with open(sense_clusters_path, 'r', encoding='utf8') as scp:
        sense_clusters = json.load(scp)

    # Read-in the attractors table
    if attractors_path is not None:
        logging.info('Reading-in known attractors ...')
        with open(attractors_path, 'r', encoding='utf8') as atp:
            attractors_table = json.load(atp)

    # Retrieve or create sense-to-cluster mappings, target sense cluster centroids
    sense_to_cluster_path = sense_clusters_path[:-5] + '_s2c.json'
    assert os.path.isfile(sense_clusters_path), 'Could not find sense-to-cluster mappings'
    if os.path.isfile(sense_to_cluster_path):
        with open(sense_to_cluster_path, 'r', encoding='utf8') as s2c:
            sense_to_cluster_table = json.load(s2c)

    # Post-process sense-to-cluster table
    sense_lemmas_to_cluster_table = None
    if is_nouns_only:
        sense_lemmas_to_cluster_table = sense_to_cluster_table
        sense_tokens_to_cluster_table = dict()
        for src_term in sense_lemmas_to_cluster_table.keys():
            sense_tokens_to_cluster_table[src_term] = dict()
            for cluster_tuple_list in sense_lemmas_to_cluster_table[src_term].values():
                for cluster_tuple in cluster_tuple_list:
                    sense_tokens_to_cluster_table[src_term][cluster_tuple[0].lower()] = [cluster_tuple]
    else:
        sense_tokens_to_cluster_table = sense_to_cluster_table

    # Initialize tables for storing extracted sentence pairs
    seed_sentence_table = dict()
    sense_links = dict()
    whitelisted_tokens_with_blacklisted_lemmas = dict()

    # Re-construct the token-to-lemma mappings for polysemous nouns
    poly_nouns_lemmas = dict()
    if attractors_path is not None:
        ambiguous_terms = list(attractors_table.keys())
    else:
        ambiguous_terms = POLYSEMOUS_NOUNS
        logging.info('No attractor table found, extracting seeds based on the list of known homographs')
    if is_nouns_only:
        logging.info('Lemmatizing polysemous nouns ...')
        for n in ambiguous_terms:
            if not seed_sentence_table.get(n, None):
                seed_sentence_table[n] = dict()
            # NOTE: Trying to enforce the correct POS does nothing with spaCy
            _, _, nl, _, _, _, _, _ = _process_strings(n, src_nlp, True, False, False, False, False)
            nl = ' '.join(nl)
            if not poly_nouns_lemmas.get(nl, None):
                poly_nouns_lemmas[nl] = n
            else:
                logging.info('Noun lemma entry clash! Lemma: {:s} | Previous value: {:s} | New value: {:s}'
                             .format(nl, poly_nouns_lemmas[nl], n))
    # Populate the word context table
    else:
        for w in sense_clusters.keys():
            if not seed_sentence_table.get(w, None):
                seed_sentence_table[w] = dict()

    # Pre-process blacklisted target senses
    logging.info('Building look-up table of whitelisted senses with blacklisted lemmas  ...')
    for term in sense_clusters.keys():
        blacklisted_target_lemmas = list()
        whitelisted_tokens_with_blacklisted_lemmas[term] = list()
        for cluster in sense_clusters[term].keys():
            whitelisted_senses = sense_clusters[term][cluster]['[SENSES]']
            blacklisted_senses = list()
            blacklisted_senses += sense_clusters[term][cluster].get('[BLACKLISTED SENSES]', [])
            blacklisted_senses += sense_clusters[term][cluster].get('[AMBIGUOUS SENSES]', [])
            if len(blacklisted_senses) > 0:
                for bl_sense in blacklisted_senses:
                    _, _, lemmas, _, _, _, _, _ = _process_strings(bl_sense, tgt_nlp, True, False, False, False, False)
                    if len(lemmas) < 1:
                        continue
                    bl_lemma = lemmas[0]
                    if bl_lemma not in blacklisted_target_lemmas:
                        blacklisted_target_lemmas.append(bl_lemma)
                for wl_sense in whitelisted_senses:
                    _, _, lemmas, _, _, _, _, _ = _process_strings(wl_sense, tgt_nlp, True, False, False, False, False)
                    if len(lemmas) < 1:
                        continue
                    wl_lemma = lemmas[0]
                    if wl_lemma in blacklisted_target_lemmas:
                        wl_sense = wl_sense.lower().strip(punctuation_plus_space)
                        if wl_sense not in whitelisted_tokens_with_blacklisted_lemmas[term]:
                            whitelisted_tokens_with_blacklisted_lemmas[term].append(wl_sense)

    logging.info('Searching bilingual corpus for sentences containing the polysemous words ... ')
    # Open files
    src_fo = open(source_file_path, 'r', encoding='utf8')
    tgt_fo = open(target_file_path, 'r', encoding='utf8')
    align_fo = None
    if alignments_file_path:
        align_fo = open(alignments_file_path, 'r', encoding='utf8')

    # Search corpora
    line_id = 0
    for line_id, src_line in enumerate(src_fo):
        # Read from files
        tgt_line = tgt_fo.readline()
        align_line = None
        if align_fo:
            align_line = align_fo.readline()
        # Lemmatize source line
        spacy_rep, spacy_src_tokens, spacy_src_lemmas, spacy_src_pos, ws_src_tokens, ws_src_tokens_lower, _, \
            spacy_to_ws_src_map = _process_strings(src_line, src_nlp, True, True, remove_stopwords, True, True)

        # Check if line contains polysemous words
        for slt_id, slt in enumerate(spacy_src_tokens):
            sll = spacy_src_lemmas[slt_id]
            # Skip stopwords
            if slt == '<STPWRD>':
                continue

            # Strip leading and trailing punctuation
            slt = slt.strip().strip(punctuation).strip()
            sll = sll.strip().strip(punctuation).strip()

            # Check if source word is ambiguous
            if is_nouns_only:
                is_ambiguous = poly_nouns_lemmas.get(sll, None)  # match lemmas
            else:
                is_ambiguous = slt in ambiguous_terms  # match tokens
            if not is_ambiguous:
                continue
            if is_ambiguous and is_nouns_only and filter_by_pos:
                if spacy_src_pos[slt_id] not in ['NOUN']:
                    continue

            # Check which relative occurrence of the slt token is being processed currently
            relative_slt_position = -1
            for tok in spacy_src_tokens[:slt_id + 1]:
                if tok == slt:
                    relative_slt_position += 1

            # Lookup possible target senses in the bilingual dictionary
            if is_nouns_only:
                sense_lemmas_to_cluster = sense_lemmas_to_cluster_table.get(poly_nouns_lemmas[sll], None)
                sense_tokens_to_cluster = sense_tokens_to_cluster_table.get(poly_nouns_lemmas[sll], None)
                whitelisted_tokens = whitelisted_tokens_with_blacklisted_lemmas.get(poly_nouns_lemmas[sll], None)
            else:
                sense_lemmas_to_cluster = None
                sense_tokens_to_cluster = sense_tokens_to_cluster_table.get(slt, None)
                whitelisted_tokens = whitelisted_tokens_with_blacklisted_lemmas.get(slt, None)

            # Check if the polysemous term has been filtered out
            if sense_lemmas_to_cluster is None and sense_tokens_to_cluster is None:
                continue

            # Lookup whitespace token locations for the current spacy token position
            ws_src_loc_list = spacy_to_ws_src_map[slt_id]
            ws_tgt_sense_loc = None
            spacy_tgt_sense_loc = None

            # Lookup target sense for the identified polysemous noun
            target_hits = list()
            # Lemmatize target line
            _, spacy_tgt_tokens, spacy_tgt_lemmas, spacy_tgt_pos, ws_tgt_tokens, ws_tgt_tokens_lower, \
                ws_to_spacy_tgt_map, spacy_to_ws_tgt_map = \
                _process_strings(tgt_line, tgt_nlp, True, True, remove_stopwords, True, True)

            # If alignments are available, lookup aligned sense and cross-reference against dictionary
            aligned_tgt_tokens = None
            matched_tgt_token = None
            is_blacklisted = False
            if align_line:
                # Convert alignments into a more practical format, first
                # NOTE: Alignments are derived from whitespace-tokenized input lines
                line_align_table = dict()
                for word_pair in align_line.strip().split():
                    ws_src_id, ws_tgt_id = word_pair.split('-')
                    ws_src_id = int(ws_src_id)
                    ws_tgt_id = int(ws_tgt_id)
                    if not line_align_table.get(ws_src_id, None):
                        line_align_table[ws_src_id] = [ws_tgt_id]
                    else:
                        line_align_table[ws_src_id].append(ws_tgt_id)

                # Map current spacy token position to the corresponding whitespace token position
                for ws_src_loc in ws_src_loc_list:

                    assert slt == ws_src_tokens_lower[ws_src_loc] or slt in ws_src_tokens_lower[ws_src_loc], \
                        'Source token mismatch between spacy \'{}\' and whitespace \'{}\'' \
                        .format(slt, ws_src_tokens_lower[ws_src_loc])

                    if line_align_table.get(ws_src_loc, None) is not None:
                        # Extend alignment window
                        line_align_table_entry = sorted(line_align_table[ws_src_loc])
                        if ATTN_WINDOW_RANGE > 0:
                            min_tgt_idx = max(0, line_align_table_entry[0] - ATTN_WINDOW_RANGE)
                            max_tgt_idx = min(len(ws_tgt_tokens) - 1, line_align_table_entry[-1] + ATTN_WINDOW_RANGE)
                            # Map from target whitespace to target spacy tokens
                            min_tgt_idx = ws_to_spacy_tgt_map[min_tgt_idx][0]
                            max_tgt_idx = ws_to_spacy_tgt_map[max_tgt_idx][0]
                            # Look-up aligned lemmas
                            tgt_window = range(min_tgt_idx, max_tgt_idx)
                        else:
                            tgt_window = line_align_table_entry
                        aligned_tgt_tokens = [spacy_tgt_tokens[idx] for idx in tgt_window]
                        aligned_tgt_lemmas = [spacy_tgt_lemmas[idx] for idx in tgt_window]
                        aligned_tgt_pos = [spacy_tgt_pos[idx] for idx in tgt_window]
                        for atl_id, atl in enumerate(aligned_tgt_lemmas):
                            # Skip stopwords
                            if atl == '<STPWRD>':
                                continue
                            # Filter by POS
                            if is_nouns_only and filter_by_pos:
                                if aligned_tgt_pos[atl_id] not in ['NOUN', 'PROPN']:
                                    continue
                            # Strip leading and trailing punctuation
                            atl = atl.strip(punctuation_plus_space)
                            # Match lemmas
                            if sense_lemmas_to_cluster is not None:
                                if sense_lemmas_to_cluster.get(atl, None):
                                    target_hits = sense_lemmas_to_cluster[atl]
                                else:
                                    maybe_lemma_matches = \
                                        sorted(sense_lemmas_to_cluster.keys(), reverse=False, key=lambda x: len(x))
                                    for maybe_lemma_match in maybe_lemma_matches:
                                        if atl.endswith(maybe_lemma_match) or atl[:-1].endswith(maybe_lemma_match):
                                            target_hits = sense_lemmas_to_cluster[maybe_lemma_match]
                                            break
                            # Match tokens
                            if len(target_hits) == 0 and sense_tokens_to_cluster is not None:
                                att = aligned_tgt_tokens[atl_id].strip(punctuation_plus_space)
                                if sense_tokens_to_cluster.get(att, None):
                                    target_hits = sense_tokens_to_cluster[att]
                                else:
                                    maybe_token_matches = \
                                        sorted(sense_tokens_to_cluster.keys(), reverse=False, key=lambda x: len(x))
                                    for maybe_token_match in maybe_token_matches:
                                        if att.endswith(maybe_token_match) or att[:-1].endswith(maybe_token_match):
                                            target_hits = sense_tokens_to_cluster[maybe_token_match]
                                            break
                            if len(target_hits) > 0:
                                spacy_tgt_sense_loc = tgt_window[atl_id]
                                ws_tgt_sense_loc = spacy_to_ws_tgt_map[spacy_tgt_sense_loc]
                                matched_tgt_token = aligned_tgt_tokens[atl_id]
                                # Check if blacklisted
                                for hit in target_hits:
                                    if hit[-1] is False:
                                        is_blacklisted = True
                                        break
                                if is_blacklisted:
                                    # Check if whitelisted
                                    for wl_token in whitelisted_tokens:
                                        if matched_tgt_token.lower().strip(punctuation_plus_space).endswith(wl_token) \
                                                or matched_tgt_token.lower().strip(punctuation_plus_space)[:-1]\
                                                .endswith(wl_token):
                                            target_hits = [hit for hit in target_hits if hit[0].lower() == wl_token]
                                            # Sanity check
                                            if len(target_hits) > 0:
                                                is_blacklisted = False
                                                break
                                if is_blacklisted:
                                    target_hits = []
                                else:
                                    break

            # Back-off to matching uninformed by alignments
            if len(target_hits) == 0 and not is_blacklisted:
                for stl_id, stl in enumerate(spacy_tgt_lemmas):
                    # Assume roughly linear source-target alignment, restrict location of possible matches
                    if stl_id not in range(slt_id - 3, slt_id + 4):
                        continue
                    # Skip stopwords
                    if stl == '<STPWRD>':
                        continue
                    # Strip leading and trailing punctuation
                    stl = stl.strip(punctuation_plus_space)
                    # Filter by POS
                    if is_nouns_only and filter_by_pos:
                        if spacy_tgt_pos[stl_id] not in ['NOUN', 'PROPN']:
                            continue
                    # Match lemmas
                    if sense_lemmas_to_cluster is not None:
                        if sense_lemmas_to_cluster.get(stl, None):
                            target_hits = sense_lemmas_to_cluster[stl]
                        else:
                            maybe_lemma_matches = \
                                sorted(sense_lemmas_to_cluster.keys(), reverse=False, key=lambda x: len(x))
                            for maybe_lemma_match in maybe_lemma_matches:
                                if stl.endswith(maybe_lemma_match) or stl[:-1].endswith(maybe_lemma_match):
                                    target_hits = sense_lemmas_to_cluster[maybe_lemma_match]
                                    break
                    # Match tokens
                    if len(target_hits) == 0 and sense_tokens_to_cluster is not None:
                        stt = spacy_tgt_tokens[stl_id].strip(punctuation_plus_space)
                        if sense_tokens_to_cluster.get(stt, None):
                            target_hits = sense_tokens_to_cluster[stt]
                        else:
                            maybe_token_matches = \
                                sorted(sense_tokens_to_cluster.keys(), reverse=False, key=lambda x: len(x))
                            for maybe_token_match in maybe_token_matches:
                                if stt.endswith(maybe_token_match) or stt[:-1].endswith(maybe_token_match):
                                    target_hits = sense_tokens_to_cluster[maybe_token_match]
                                    break
                    if len(target_hits) > 0:
                        spacy_tgt_sense_loc = stl_id
                        ws_tgt_sense_loc = spacy_to_ws_tgt_map[spacy_tgt_sense_loc]
                        matched_tgt_token = spacy_tgt_tokens[stl_id]
                        # Check if blacklisted
                        for hit in target_hits:
                            if hit[-1] is False:
                                is_blacklisted = True
                                break
                        if is_blacklisted:
                            # Check if whitelisted
                            for wl_token in whitelisted_tokens:
                                if matched_tgt_token.lower().strip(punctuation_plus_space).endswith(wl_token) or \
                                        matched_tgt_token.lower().strip(punctuation_plus_space)[:-1].endswith(wl_token):
                                    target_hits = [hit for hit in target_hits if hit[0].lower() == wl_token]
                                    # Sanity check
                                    if len(target_hits) > 0:
                                        is_blacklisted = False
                                        break
                        if is_blacklisted:
                            target_hits = []
                        else:
                            break

            # Collect sentence pairs
            if len(target_hits) == 0:
                if verbose:
                    logging.info('REJECTED LINE:')
                    logging.info(src_line.strip())
                    logging.info(tgt_line.strip())
                    logging.info('MATCHED SOURCE TERM: {:s} / {:s}'.format(slt, sll))
                    logging.info('MATCHED TGT TOKEN: {}'.format(matched_tgt_token))
                    logging.info('ALIGNED TGT TOKENS: {}'.format(aligned_tgt_tokens))
                    logging.info('LEMMAS and POS: {}'.format(list(zip(spacy_tgt_lemmas, spacy_tgt_pos))))
                    logging.info('-' * 20)
                continue
            else:
                # Pick the right hit
                hit_clusters = list(set([hit[1] for hit in target_hits]))
                if len(hit_clusters) > 1:
                    logging.info('Multiple target senses matched to the source homograph! {}, {}, {}'
                                 .format(src_line, tgt_line, target_hits))
                    continue
                tgt_assignment = None
                target_hits = sorted(target_hits, reverse=True, key=lambda x: len(x[0]))
                for hit in target_hits:
                    if hit[0].lower() in matched_tgt_token.lower():
                        tgt_assignment = hit
                if tgt_assignment is None:
                    tgt_assignment = (matched_tgt_token, target_hits[0][1], target_hits[0][2])

                # Extend seed sentence table
                if is_nouns_only:
                    if not seed_sentence_table.get(poly_nouns_lemmas[sll], None):
                        seed_sentence_table[poly_nouns_lemmas[sll]] = dict()
                    if merge_source_forms:
                        src_table_entry = seed_sentence_table[poly_nouns_lemmas[sll]]
                    else:
                        if not seed_sentence_table[poly_nouns_lemmas[sll]].get(slt, None):
                            seed_sentence_table[poly_nouns_lemmas[sll]][slt] = dict()
                        src_table_entry = seed_sentence_table[poly_nouns_lemmas[sll]][slt]
                else:
                    if not seed_sentence_table.get(slt, None):
                        seed_sentence_table[slt] = dict()
                    src_table_entry = seed_sentence_table[slt]

                tgt_assignment = target_hits[0]
                # Extend the table of attractors specific to a target sense cluster
                if not src_table_entry.get(tgt_assignment[1], None):
                    src_table_entry[tgt_assignment[1]] = [(src_line, tgt_line,
                                                           spacy_src_pos, spacy_tgt_pos,
                                                           align_line, slt, tgt_assignment[0],
                                                           slt_id, ws_src_loc_list,
                                                           spacy_tgt_sense_loc, ws_tgt_sense_loc)]
                else:
                    src_table_entry[tgt_assignment[1]].append((src_line, tgt_line,
                                                               spacy_src_pos, spacy_tgt_pos,
                                                               align_line, slt, tgt_assignment[0],
                                                               slt_id, ws_src_loc_list,
                                                               spacy_tgt_sense_loc, ws_tgt_sense_loc))

        # Report occasionally
        matched_src = [k for k in seed_sentence_table.keys() if len(seed_sentence_table[k].keys()) > 0]
        num_matched_sentences = list()
        if is_nouns_only and not merge_source_forms:
            for t in matched_src:
                for f in seed_sentence_table[t].keys():
                    for sc in seed_sentence_table[t][f].keys():
                        num_matched_sentences.append(len(seed_sentence_table[t][f][sc]))
        else:
            for t in matched_src:
                for sc in seed_sentence_table[t].keys():
                    num_matched_sentences.append(len(seed_sentence_table[t][sc]))

        if line_id > 0 and line_id % 1000 == 0:
            logging.info('== Running statistics for seed sentence extraction ==')
            logging.info('Read-in {:d} lines | Created entries for {:d} homographs'
                         .format(line_id, len(matched_src)))
            logging.info('Found {:d} sentence pairs; ({:.3f} per item on average)'
                         .format(sum(num_matched_sentences), float(np.mean(num_matched_sentences))))
            logging.info('-' * 20)

    # Final report
    matched_src = [k for k in seed_sentence_table.keys() if len(seed_sentence_table[k].keys()) > 0]
    num_matched_sentences = list()
    if is_nouns_only and not merge_source_forms:
        for t in matched_src:
            for f in seed_sentence_table[t].keys():
                for sc in seed_sentence_table[t][f].keys():
                    num_matched_sentences.append(len(seed_sentence_table[t][f][sc]))
    else:
        for t in matched_src:
            for sc in seed_sentence_table[t].keys():
                num_matched_sentences.append(len(seed_sentence_table[t][sc]))
    logging.info('== FINAL statistics for seed sentence extraction ==')
    logging.info('Read-in {:d} lines | Created entries for {:d} homographs'
                 .format(line_id, len(matched_src)))
    logging.info('Found {:d} sentences; ({:.3f} per item on average)'
                 .format(sum(num_matched_sentences), float(np.mean(num_matched_sentences))))
    logging.info('-' * 20)

    # Save seed sentence pairs
    file_type = '_'.join(sense_clusters_path.split('/')[-1].split('_')[3:5])
    logging.info('Saving extracted seed sentence pairs ...')
    seed_sentences_path = out_file_path[:-5] + '_{:s}.json'.format(file_type)
    with open(seed_sentences_path, 'w', encoding='utf8') as ssp:
        json.dump(seed_sentence_table, ssp, indent=3, sort_keys=True, ensure_ascii=False)

    # Saving sense links
    sense_links_path = out_file_path[:-5] + '_sense_links_{:s}.json'.format(file_type)
    with open(sense_links_path, 'w', encoding='utf8') as slp:
        json.dump(sense_links, slp, indent=3, sort_keys=True, ensure_ascii=False)
    logging.info('Saved the seed sentence pairs to {:s}'.format(out_file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file_path', type=str, required=True,
                        help='path to file containing the source side of the extraction corpus')
    parser.add_argument('--target_file_path', type=str, required=True,
                        help='path to file containing the target side of the extraction corpus')
    parser.add_argument('--alignments_file_path', type=str, default=None,
                        help='path to file containing the fastalign alignments for the extraction corpus')
    parser.add_argument('--sense_clusters_path', type=str, default=None,
                        help='path to the JSON file containing scraped BabelNet sense clusters')
    parser.add_argument('--attractors_path', type=str, default=None,
                        help='path to the JSON file containing the extracted attractor terms')
    parser.add_argument('--out_file_path', type=str, required=True,
                        help='path to which the seed sentence pairs should be saved')
    parser.add_argument('--filter_by_pos', action='store_true', help='limits context words to (proper) nouns')
    parser.add_argument('--remove_stopwords', action='store_true',
                        help='toggles the exclusion of stopwords from source and target sentences '
                             '(and thus from the attractor set)')
    parser.add_argument('--ignore_ambiguous_senses', action='store_true',
                        help='determines whether sentence pairs containing ambiguous target sense terms are excluded '
                             'from the seed sentences set')
    parser.add_argument('--lang_pair', type=str, default=None,
                        help='language pair of the bitext; expected format is src-tgt')
    parser.add_argument('--verbose', action='store_true', help='enables additional logging used for debugging')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.out_file_path.split('/')[:-1])
    file_name = args.out_file_path.split('/')[-1]
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

    collect_seed_samples(args.source_file_path,
                         args.target_file_path,
                         args.alignments_file_path,
                         args.sense_clusters_path,
                         args.attractors_path,
                         args.out_file_path,
                         args.filter_by_pos,
                         args.remove_stopwords,
                         args.verbose)

