import os
import re
import sys
import json
import spacy
import string
import logging
import argparse

import numpy as np
from nltk.corpus import stopwords


WORD_ATTRACTOR_TAGS = ['ADJ']
COMPOUND_POS_LIST = ['NOUN', 'PROPN']
BLACKLIST = ['@-@']
SPECIAL_SYMBOLS = ['&apos;', '&quot;']
# From:
# https://dictionary.cambridge.org/grammar/british-grammar/much-many-a-lot-of-lots-of-quantifiers
# https://www.myenglishpages.com/site_php_files/grammar-lesson-quantifiers.php
QUANTIFIERS = ['much', 'many', 'lot', 'lots', 'few', 'bit', 'all', 'more', 'most', 'enough', 'less', 'least', 'any',
               'none', 'some', 'plenty', 'several', ]
MODIFIERS_POS_SET = ['NOUN', 'VERB', 'ADJ']
CONTRACTIONS = \
    ['t', 've', 'd', 'ye', 'e', 'er', 's', 'g', 'n', 'll', 're', 'm', 'a', 'o', 're', 'y', 'gon', 'wan', 'na', '\'']
VOWELS = 'aeiou'
CONSONANTS = 'bcdfghjklmnpqrstvwxyz'
INTENSIFIERS = ['damn', 'fucking', 'freaking', 'flipping']
ORDINAL_SUFFIXES = ['first', 'second', 'third', 'th']


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


def _insert_attractor_at_homograph(src_sent,
                                   src_rep,
                                   src_rep_loc,
                                   attractor_term,
                                   attractor_cluster_id,
                                   filter_bigrams,
                                   seen_samples,
                                   seed_parses,
                                   disable_ngrams):

    """ Generates adversarial samples from a single seed sentence by inserting the attractor term at each position
    within a pre-defined window around the ambiguous term. """

    # Track samples to reduce duplicates
    orig_seen_samples = seen_samples
    # Process source sequence
    spacy_sent_rep = seed_parses[src_sent][0]
    spacy_tokens_lower = seed_parses[src_sent][1]
    ws_tokens = seed_parses[src_sent][2]
    ws_tokens_lower = seed_parses[src_sent][3]
    spacy_to_ws_map = seed_parses[src_sent][4]

    spacy_src_rep_ids = list()
    adversarial_samples = list()

    # Filter adj + noun with monolingual bigrams
    if not disable_ngrams:
        if filter_bigrams.get(attractor_term, None) is not None and attractor_term not in INTENSIFIERS:
            modified_term = src_rep.lower().strip(punctuation_plus_space)
            bigram_count = filter_bigrams[attractor_term].get(modified_term, 0.)
            if bigram_count < 10:
                if attractor_term.endswith('er'):
                    if filter_bigrams.get(attractor_term[:-2], None) is not None:
                        bigram_count = filter_bigrams[attractor_term[:-2]].get(modified_term, 0.)
                if attractor_term.endswith('est'):
                    if filter_bigrams.get(attractor_term[:-3], None) is not None:
                        bigram_count = filter_bigrams[attractor_term[:-3]].get(modified_term, 0.)
                if bigram_count < 10:
                    return adversarial_samples, orig_seen_samples

    # Ignore (most) comparative / superlative attractor terms
    if attractor_term.endswith('er') or attractor_term.endswith('est'):
        # Count non-consecutive vowels
        vowel_seq = list()
        for ch in attractor_term:
            if ch in VOWELS:
                if len(vowel_seq) == 0:
                    vowel_seq.append(1)
                else:
                    if vowel_seq[-1] != 1:
                        vowel_seq.append(1)
            else:
                vowel_seq.append(0)
        if sum(vowel_seq) == 2:
            return adversarial_samples, orig_seen_samples

    # Detect appropriate positions
    for token_id, token in enumerate(spacy_tokens_lower):
        # Remove punctuation, separate compounds
        sub_token_list = re.sub(r' +', ' ', token.translate(pct_stripper)).split()
        sub_token_list = [sub_token.strip(punctuation_plus_space) for sub_token in sub_token_list]
        for sub_token in sub_token_list:
            if sub_token == src_rep:
                spacy_src_rep_ids.append(token_id)
                break  # only one sub-token hit per token allowed

    if len(spacy_src_rep_ids) == 0:
        return adversarial_samples, orig_seen_samples
    else:
        attractor_len = len(attractor_term.split())
        insertion_site = src_rep_loc
        do_insert = False
        right_context = spacy_sent_rep[insertion_site]
        if right_context.pos_ in ['NOUN', 'PROPN'] and \
                right_context.text.lower().strip(punctuation_plus_space) != attractor_term:
            do_insert = True
        # Avoid inserting attractors in the middle of compounds
        # Insertion next to adjectives is prohibited due to adjectival order in English
        if do_insert and insertion_site > 0:
            left_context = spacy_sent_rep[insertion_site - 1]
            if left_context.pos_ in COMPOUND_POS_LIST or spacy_sent_rep[insertion_site].text in CONTRACTIONS:
                do_insert = False
            if left_context.pos_ == 'ADJ' or left_context.text == '@-@':
                do_insert = False
            if insertion_site > 1:
                if spacy_sent_rep[insertion_site - 2].text in ['a', 'an', 'the']:
                    do_insert = False
        if do_insert and insertion_site < (len(spacy_sent_rep) - 1):
            # Avoid modifying nouns followed by apostrophes
            if spacy_sent_rep[insertion_site + 1].text.startswith('\''):
                do_insert = False
            # Avoid modifying compounds (e.g. 'arm strength')
            if spacy_sent_rep[insertion_site + 1].pos_ in ['NOUN', 'PROPN']:
                do_insert = False
        if not do_insert:
            return adversarial_samples, orig_seen_samples

        # Determine position of ambiguous term in whitespace tokens
        ws_src_rep_loc = spacy_to_ws_map[src_rep_loc][0]
        ws_insertion_site = spacy_to_ws_map[insertion_site][0]
        try:
            assert spacy_tokens_lower[insertion_site] == ws_tokens_lower[spacy_to_ws_map[insertion_site][0]] or \
                spacy_tokens_lower[insertion_site] in ws_tokens_lower[spacy_to_ws_map[insertion_site][0]], \
                'Spacy token at attractor insertion site {} | {} does not match the corresponding whitespace token ' \
                '{} | {}'.format(spacy_tokens_lower[insertion_site], spacy_tokens_lower,
                                 ws_tokens_lower[spacy_to_ws_map[insertion_site][0]], ws_tokens_lower)
        except AssertionError as error:
            ignore = False
            for ss in SPECIAL_SYMBOLS:
                if ws_tokens_lower[spacy_to_ws_map[insertion_site][0]].startswith(ss):
                    ignore = True
                    break
            if not ignore:
                raise AssertionError(error)

        # Account for a / an
        if ws_insertion_site > 0:
            if ws_tokens[ws_insertion_site - 1] == 'a':
                for vowel in list(VOWELS):
                    if attractor_term.startswith(vowel):
                        ws_tokens[ws_insertion_site - 1] = 'an'
            if ws_tokens[ws_insertion_site - 1] == 'an':
                for consonant in list(CONSONANTS):
                    if attractor_term.startswith(consonant):
                        ws_tokens[ws_insertion_site - 1] = 'a'

        # Account for superlatives and ordinals
        change_det = False
        for suffix in ['est'] + ORDINAL_SUFFIXES:
            if attractor_term.endswith(suffix):
                change_det = True
                break
        if change_det:
            if ws_insertion_site > 0:
                if ws_tokens[ws_insertion_site - 1] in ['a', 'an']:
                    ws_tokens[ws_insertion_site - 1] = 'the'

        # Generate samples
        new_sent_tokens = ws_tokens[:ws_insertion_site] + [attractor_term] + ws_tokens[ws_insertion_site:]
        new_sent = ' '.join(new_sent_tokens)
        attractor_ws_ids = [ws_insertion_site + attr_tok_id for attr_tok_id in range(attractor_len)]
        updated_ambiguous_term_ws_ids = list()
        updated_ambiguous_focus_term_ws_id = ws_src_rep_loc
        for rep_id in spacy_src_rep_ids:
            updated_rep_id = spacy_to_ws_map[rep_id][0]
            if updated_rep_id >= ws_insertion_site:
                updated_rep_id = updated_rep_id + attractor_len
                if rep_id == src_rep_loc:
                    updated_ambiguous_focus_term_ws_id = updated_rep_id
            updated_ambiguous_term_ws_ids.append(updated_rep_id)

        assert ws_tokens[spacy_to_ws_map[src_rep_loc][0]] == new_sent_tokens[updated_ambiguous_focus_term_ws_id], \
            'Mismatch between token at ambiguous token position in the original sentence \'{}\' | \'{}\' ' \
            'and generated sample \'{}\' | \'{}\''.format(src_sent.strip(), spacy_to_ws_map[src_rep_loc][0],
                                                          new_sent, updated_ambiguous_focus_term_ws_id)
        assert updated_ambiguous_focus_term_ws_id in updated_ambiguous_term_ws_ids, \
            'Term ID adjustment mismatch: Focus term ID: {}, ambiguous term IDs: {}' \
            .format(updated_ambiguous_focus_term_ws_id, updated_ambiguous_term_ws_ids)

        # Check if duplicate
        if seen_samples.get(new_sent, None):
            if seen_samples[new_sent] == (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id):
                return adversarial_samples, orig_seen_samples
        else:
            seen_samples[new_sent] = (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id)
            adversarial_samples.append((new_sent,
                                        updated_ambiguous_term_ws_ids,
                                        updated_ambiguous_focus_term_ws_id,
                                        attractor_ws_ids))
    return adversarial_samples, seen_samples


def _insert_attractor_at_other_nouns(src_sent,
                                     src_rep,
                                     src_rep_loc,
                                     attractor_term,
                                     attractor_table,
                                     seed_attractor_tokens,
                                     adversarial_attractor_tokens,
                                     general_modifier_tokens,
                                     general_modifier_lemmas,
                                     filter_bigrams,
                                     window_size,
                                     seen_samples,
                                     attractor_cluster_id,
                                     seed_parses,
                                     disable_modifiers,
                                     disable_ngrams,
                                     pos_list=('NOUN')):

    """ Generates adversarial samples from a single seed sentence by inserting the attractor term next to every
    token that shares the POS with the ambiguous word (excluding the ambiguous word itself). """
    # Track samples to reduce duplicates
    orig_seen_samples = seen_samples
    # Process source sequence
    spacy_sent_rep = seed_parses[src_sent][0]
    spacy_tokens_lower = seed_parses[src_sent][1]
    ws_tokens = seed_parses[src_sent][2]
    ws_tokens_lower = seed_parses[src_sent][3]
    spacy_to_ws_map = seed_parses[src_sent][4]

    spacy_src_rep_ids = list()
    spacy_rel_pos_ids = list()
    adversarial_samples = list()

    # Ignore (most) comparative / superlative attractor terms
    if attractor_term.endswith('er') or attractor_term.endswith('est'):
        # Count non-consecutive vowels
        vowel_seq = list()
        for ch in attractor_term:
            if ch in VOWELS:
                if len(vowel_seq) == 0:
                    vowel_seq.append(1)
                else:
                    if vowel_seq[-1] != 1:
                        vowel_seq.append(1)
            else:
                vowel_seq.append(0)
        if sum(vowel_seq) == 2:
            return adversarial_samples, orig_seen_samples

    # Detect appropriate positions
    for token_id, token in enumerate(spacy_tokens_lower):
        if token in BLACKLIST:
            continue
        # Remove punctuation, separate compounds
        sub_token_list = re.sub(r' +', ' ', token.translate(pct_stripper)).split()
        sub_token_list = [sub_token.strip(punctuation_plus_space) for sub_token in sub_token_list]
        for sub_token in sub_token_list:
            if sub_token == src_rep:
                spacy_src_rep_ids.append(token_id)
                break  # only one sub-token hit per token allowed
        if token_id not in spacy_src_rep_ids and spacy_sent_rep[token_id].pos_ in pos_list:
            if token_id not in spacy_rel_pos_ids:
                if token_id < len(spacy_sent_rep) - 1:
                    if spacy_sent_rep[token_id + 1].text.startswith('\''):
                        continue
                spacy_rel_pos_ids.append(token_id)

    if len(spacy_src_rep_ids) == 0 or len(spacy_rel_pos_ids) == 0:
        return adversarial_samples, orig_seen_samples
    else:
        # Restrict set of modified terms to a window around each occurrence of the ambiguous term
        if len(spacy_rel_pos_ids) > window_size > 0:
            truncated_spacy_rel_pos_ids = list()
            truncated_spacy_rel_pos_ids += sorted(spacy_rel_pos_ids, key=lambda x: abs(x-src_rep_loc))[:window_size]
            spacy_rel_pos_ids = list(set(truncated_spacy_rel_pos_ids))

        for token_id in spacy_rel_pos_ids:
            if not disable_modifiers:
                # Check if attractor is permissible
                noun_rep = spacy_sent_rep[token_id]
                noun_lemma = noun_rep.lower_ if noun_rep.lemma_ == '-PRON-' or \
                    noun_rep.lemma_.isdigit() else noun_rep.lemma_.lower()
                noun_lemma = noun_lemma.strip(punctuation_plus_space)
                modifier_tokens = general_modifier_tokens.get(noun_lemma, None)
                modifier_lemmas = general_modifier_lemmas.get(noun_lemma, None)

                if modifier_tokens is None:
                    continue
                else:
                    # Check whether the modified noun should be modified by the current attractor
                    keep_attractor_term = _score_attractor_with_modifiers(attractor_term,
                                                                          attractor_table,
                                                                          modifier_tokens,
                                                                          modifier_lemmas,
                                                                          seed_attractor_tokens,
                                                                          adversarial_attractor_tokens)
                    if not keep_attractor_term:
                        continue

            # Filter with bigrams
            if not disable_ngrams:
                if filter_bigrams.get(attractor_term, None) is not None and attractor_term not in INTENSIFIERS:
                    modified_term = src_rep.lower().strip(punctuation_plus_space)
                    bigram_count = filter_bigrams[attractor_term].get(modified_term, 0.)
                    if bigram_count < 10:
                        if attractor_term.endswith('er'):
                            if filter_bigrams.get(attractor_term[:-2], None) is not None:
                                bigram_count = filter_bigrams[attractor_term[:-2]].get(modified_term, 0.)
                        if attractor_term.endswith('est'):
                            if filter_bigrams.get(attractor_term[:-3], None) is not None:
                                bigram_count = filter_bigrams[attractor_term[:-3]].get(modified_term, 0.)
                        if bigram_count < 10:
                            return adversarial_samples, orig_seen_samples

            # Avoid inserting attractors in the middle of compounds
            attractor_len = len(attractor_term.split())
            insertion_site = token_id
            do_insert = False
            right_context = spacy_sent_rep[insertion_site]
            if right_context.pos_ in ['NOUN', 'PROPN'] and \
                    right_context.text.lower().strip(punctuation_plus_space) != attractor_term:
                do_insert = True
            # Avoid inserting attractors in the middle of compounds
            # Insertion next to adjectives is prohibited due to adjectival order in English
            if do_insert and insertion_site > 0:
                left_context = spacy_sent_rep[insertion_site - 1]
                if left_context.pos_ in COMPOUND_POS_LIST or spacy_sent_rep[insertion_site].text in CONTRACTIONS:
                    do_insert = False
                if left_context.pos_ == 'ADJ' or left_context.text == '@-@':
                    do_insert = False
                if insertion_site > 1:
                    if spacy_sent_rep[insertion_site - 2].text in ['a', 'an', 'the']:
                        do_insert = False
            if do_insert and insertion_site < (len(spacy_sent_rep) - 1):
                # Avoid modifying nouns followed by apostrophes
                if spacy_sent_rep[insertion_site + 1].text.startswith('\''):
                    do_insert = False
                # Avoid modifying compounds (e.g. 'arm strength')
                if spacy_sent_rep[insertion_site + 1].pos_ in ['NOUN', 'PROPN']:
                    do_insert = False
            if not do_insert:
                continue

            # Determine position of ambiguous term in whitespace tokens
            ws_src_rep_loc = spacy_to_ws_map[src_rep_loc][0]
            ws_insertion_site = spacy_to_ws_map[insertion_site][0]
            try:
                assert spacy_tokens_lower[insertion_site] == ws_tokens_lower[spacy_to_ws_map[insertion_site][0]] or \
                    spacy_tokens_lower[insertion_site] in ws_tokens_lower[spacy_to_ws_map[insertion_site][0]], \
                    'Spacy token at attractor insertion site {} | {} does not match the corresponding whitespace ' \
                    'token {} | {}'.format(spacy_tokens_lower[insertion_site], spacy_tokens_lower,
                                           ws_tokens_lower[spacy_to_ws_map[insertion_site][0]], ws_tokens_lower)
            except AssertionError as error:
                ignore = False
                for ss in SPECIAL_SYMBOLS:
                    if ws_tokens_lower[spacy_to_ws_map[insertion_site][0]].startswith(ss):
                        ignore = True
                        break
                if not ignore:
                    raise AssertionError(error)

            # Account for a / an
            if ws_insertion_site > 0:
                if ws_tokens[ws_insertion_site - 1] == 'a':
                    for vowel in list(VOWELS):
                        if attractor_term.startswith(vowel):
                            ws_tokens[ws_insertion_site - 1] = 'an'
                if ws_tokens[ws_insertion_site - 1] == 'an':
                    for consonant in list(CONSONANTS):
                        if attractor_term.startswith(consonant):
                            ws_tokens[ws_insertion_site - 1] = 'a'

            # Account for superlatives and ordinals
            change_det = False
            for suffix in ['est'] + ORDINAL_SUFFIXES:
                if attractor_term.endswith(suffix):
                    change_det = True
                    break
            if change_det:
                if ws_insertion_site > 0:
                    if ws_tokens[ws_insertion_site - 1] in ['a', 'an']:
                        ws_tokens[ws_insertion_site - 1] = 'the'

            # Generate samples by inserting the attractor in the neighborhood of each token of the appropriate POS
            new_sent_tokens = ws_tokens[:ws_insertion_site] + [attractor_term] + ws_tokens[ws_insertion_site:]
            new_sent = ' '.join(new_sent_tokens)
            attractor_ws_ids = [ws_insertion_site + attr_tok_id for attr_tok_id in range(len(attractor_term.split()))]
            updated_ambiguous_term_ws_ids = list()
            updated_ambiguous_focus_term_ws_id = ws_src_rep_loc
            for rep_id in spacy_src_rep_ids:
                updated_rep_id = spacy_to_ws_map[rep_id][0]
                if updated_rep_id >= ws_insertion_site:
                    updated_rep_id = updated_rep_id + attractor_len
                    if rep_id == src_rep_loc:
                        updated_ambiguous_focus_term_ws_id = updated_rep_id
                updated_ambiguous_term_ws_ids.append(updated_rep_id)

            assert ws_tokens[spacy_to_ws_map[src_rep_loc][0]] == new_sent_tokens[updated_ambiguous_focus_term_ws_id], \
                'Mismatch between token at ambiguous token position in the original sentence \'{}\' | \'{}\' ' \
                'and generated sample \'{}\' | \'{}\''.format(src_sent.strip(), spacy_to_ws_map[src_rep_loc][0],
                                                              new_sent, updated_ambiguous_focus_term_ws_id)
            assert updated_ambiguous_focus_term_ws_id in updated_ambiguous_term_ws_ids, \
                'Term ID adjustment mismatch: Focus term ID: {}, ambiguous term IDs: {}' \
                .format(updated_ambiguous_focus_term_ws_id, updated_ambiguous_term_ws_ids)

            # Check if duplicate
            if seen_samples.get(new_sent, None):
                if seen_samples[new_sent] == (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id):
                    continue
            else:
                seen_samples[new_sent] = (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id)
                adversarial_samples.append((new_sent,
                                            updated_ambiguous_term_ws_ids,
                                            updated_ambiguous_focus_term_ws_id,
                                            attractor_ws_ids))
    return adversarial_samples, seen_samples


def _replace_attractor_at_homograph(src_sent,
                                    src_rep,
                                    src_rep_loc,
                                    attractor_term,
                                    attractor_cluster_id,
                                    filter_bigrams,
                                    seen_samples,
                                    seed_parses,
                                    disable_ngrams):

    """ Generates adversarial samples from a single seed sentence by replacing seed sentence tokens of the same POS
    category as the attractor term with the attractor. """

    def _is_non_positive(adjective):
        """ Helper function for checking whether the specified adjective is a comparative or superlative """
        # Count non-consecutive vowels
        vowel_seq = list()
        for ch in adjective:
            if ch in VOWELS:
                if len(vowel_seq) == 0:
                    vowel_seq.append(1)
                else:
                    if vowel_seq[-1] != 1:
                        vowel_seq.append(1)
            else:
                vowel_seq.append(0)
        if sum(vowel_seq) == 2:
            return True

    # Track samples to reduce duplicates
    orig_seen_samples = seen_samples
    # Process source sequence
    spacy_sent_rep = seed_parses[src_sent][0]
    spacy_tokens_lower = seed_parses[src_sent][1]
    ws_tokens = seed_parses[src_sent][2]
    spacy_to_ws_map = seed_parses[src_sent][4]

    spacy_src_rep_ids = list()
    spacy_rel_pos_ids = list()
    adversarial_samples = list()
    tokens_to_modify = list()

    # Filter with bigrams
    if not disable_ngrams:
        if filter_bigrams.get(attractor_term, None) is not None and attractor_term not in INTENSIFIERS:
            modified_term = src_rep.lower().strip(punctuation_plus_space)
            bigram_count = filter_bigrams[attractor_term].get(modified_term, 0.)
            if bigram_count < 10:
                if attractor_term.endswith('er'):
                    if filter_bigrams.get(attractor_term[:-2], None) is not None:
                        bigram_count = filter_bigrams[attractor_term[:-2]].get(modified_term, 0.)
                if attractor_term.endswith('est'):
                    if filter_bigrams.get(attractor_term[:-3], None) is not None:
                        bigram_count = filter_bigrams[attractor_term[:-3]].get(modified_term, 0.)
                if bigram_count < 10:
                    return adversarial_samples, seen_samples

    #  Check children for adjectives to replace
    children = [child for child in spacy_sent_rep[src_rep_loc].children]
    for child in children:
        if child.pos_ == 'ADJ' and child.text not in [tpl[0] for tpl in tokens_to_modify] \
                and child.text.lower().strip(string.punctuation) not in QUANTIFIERS:
            if child.text[0] == child.text[0].lower():
                tokens_to_modify.append((child.text, child.i))

    # Check if attractor is permissible
    for adj, adj_loc in tokens_to_modify:
        if adj_loc not in spacy_rel_pos_ids:

            # Avoid breaking-up collocations
            if not disable_ngrams:
                term_to_replace = adj.lower().strip(punctuation_plus_space)
                if filter_bigrams.get(term_to_replace, None) is not None and attractor_term not in INTENSIFIERS:
                    modified_term = src_rep.lower().strip(punctuation_plus_space)
                    bigram_count = filter_bigrams[term_to_replace].get(modified_term, 0)
                    if bigram_count >= 300:  # eye-balled value
                        continue

            # Check if insertion constraints are violated
            if spacy_sent_rep[adj_loc].text.lower().strip(punctuation_plus_space) == attractor_term:
                continue
            try:
                left_context = spacy_sent_rep[adj_loc - 1]
            except IndexError:
                left_context = None
            try:
                right_context = spacy_sent_rep[adj_loc + 1]
            except IndexError:
                right_context = None
            if right_context is not None:
                if right_context.pos_ not in ['NOUN', 'PROPN'] or \
                        (right_context.text.lower().strip(punctuation_plus_space) == attractor_term):
                    continue
            if left_context is not None:
                if left_context.pos_ in ['ADJ', 'PROPN'] or left_context.text == '@-@':
                    continue
                if adj_loc > 1:
                    if spacy_sent_rep[adj_loc - 2].text in ['a', 'an', 'the']:
                        continue
            if adj_loc < (len(spacy_sent_rep) - 2):
                # Avoid modifying compounds (e.g. 'arm strength')
                if spacy_sent_rep[adj_loc + 2].pos_ in ['NOUN', 'PROPN']:
                    continue
            spacy_rel_pos_ids.append(adj_loc)

    # Detect appropriate positions
    for token_id, token in enumerate(spacy_tokens_lower):
        if token in BLACKLIST:
            continue
        # Remove punctuation, separate compounds
        sub_token_list = re.sub(r' +', ' ', token.translate(pct_stripper)).split()
        sub_token_list = [sub_token.strip(punctuation_plus_space) for sub_token in sub_token_list]
        for sub_token in sub_token_list:
            if sub_token == src_rep:
                spacy_src_rep_ids.append(token_id)
                break  # only one sub-token hit per token allowed

    if len(spacy_src_rep_ids) == 0 or len(spacy_rel_pos_ids) == 0:
        return adversarial_samples, orig_seen_samples
    else:
        attractor_len = len(attractor_term.split())
        for token_id in spacy_rel_pos_ids:
            # Convert to whitespace token position
            ws_token_id = spacy_to_ws_map[token_id][0]

            # Account for a / an
            if ws_token_id > 0:
                if ws_tokens[ws_token_id - 1] == 'a':
                    for vowel in list(VOWELS):
                        if attractor_term.startswith(vowel):
                            ws_tokens[ws_token_id - 1] = 'an'
                if ws_tokens[ws_token_id - 1] == 'an':
                    for consonant in list(CONSONANTS):
                        if attractor_term.startswith(consonant):
                            ws_tokens[ws_token_id - 1] = 'a'

            # Replace (most) adjectives with similar adjective forms
            if attractor_term.endswith('er') and _is_non_positive(attractor_term):
                if not (spacy_tokens_lower[token_id].endswith('er') and _is_non_positive(spacy_tokens_lower[token_id])):
                    continue
            if attractor_term.endswith('est') and _is_non_positive(attractor_term):
                if not (spacy_tokens_lower[token_id].endswith('est') and
                        _is_non_positive(spacy_tokens_lower[token_id])):
                    continue
            if (not (attractor_term.endswith('er') or attractor_term.endswith('est'))) or \
                    (not _is_non_positive(attractor_term)):
                if (spacy_tokens_lower[token_id].endswith('er') or spacy_tokens_lower[token_id].endswith('est')) and \
                        _is_non_positive(spacy_tokens_lower[token_id]):
                    continue

            # Account for superlatives and ordinals
            change_det = False
            for suffix in ['est'] + ORDINAL_SUFFIXES:
                if attractor_term.endswith(suffix):
                    change_det = True
                    break
            if change_det:
                if ws_token_id > 0:
                    if ws_tokens[ws_token_id - 1] in ['a', 'an']:
                        ws_tokens[ws_token_id - 1] = 'the'

            # Generate samples by inserting the attractor in the neighborhood of each token of the appropriate POS
            new_sent_tokens = ws_tokens[:ws_token_id] + [attractor_term] + ws_tokens[ws_token_id + 1:]
            new_sent = ' '.join(new_sent_tokens)
            attractor_ws_ids = [ws_token_id + attr_tok_id for attr_tok_id in range(len(attractor_term.split()))]
            updated_ambiguous_term_ws_ids = list()
            updated_ambiguous_focus_term_ws_id = spacy_to_ws_map[src_rep_loc][0]
            for rep_id in spacy_src_rep_ids:
                updated_rep_id = spacy_to_ws_map[rep_id][0]
                if updated_rep_id >= ws_token_id:
                    updated_rep_id = updated_rep_id - len(ws_tokens[ws_token_id].split()) + attractor_len
                    if rep_id == src_rep_loc:
                        updated_ambiguous_focus_term_ws_id = updated_rep_id
                updated_ambiguous_term_ws_ids.append(updated_rep_id)

            assert ws_tokens[spacy_to_ws_map[src_rep_loc][0]] == new_sent_tokens[updated_ambiguous_focus_term_ws_id], \
                'Mismatch between token at ambiguous token position in the original sentence \'{}\' | \'{}\' ' \
                'and generated sample \'{}\' | \'{}\''.format(src_sent.strip(), spacy_to_ws_map[src_rep_loc][0],
                                                              new_sent, updated_ambiguous_focus_term_ws_id)
            assert updated_ambiguous_focus_term_ws_id in updated_ambiguous_term_ws_ids, \
                'Term ID adjustment mismatch: Focus term ID: {}, ambiguous term IDs: {}' \
                .format(updated_ambiguous_focus_term_ws_id, updated_ambiguous_term_ws_ids)

            # Check if duplicate
            if seen_samples.get(new_sent, None):
                if seen_samples[new_sent] == (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id):
                    continue
            else:
                seen_samples[new_sent] = (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id)
                adversarial_samples.append((new_sent,
                                            updated_ambiguous_term_ws_ids,
                                            updated_ambiguous_focus_term_ws_id,
                                            attractor_ws_ids))
    return adversarial_samples, seen_samples


def _replace_attractor_at_other_nouns(src_sent,
                                      src_rep,
                                      src_rep_loc,
                                      attractor_term,
                                      attractor_table,
                                      seed_attractor_tokens,
                                      adversarial_attractor_tokens,
                                      general_modifier_tokens,
                                      general_modifier_lemmas,
                                      filter_bigrams,
                                      window_size,
                                      seen_samples,
                                      attractor_cluster_id,
                                      seed_parses,
                                      disable_modifiers,
                                      disable_ngrams):

    """ Generates adversarial samples from a single seed sentence by replacing seed sentence tokens of the same POS
    category as the attractor term with the attractor
    (except for cases where the seed token modifies the ambiguous noun). """

    def _is_non_positive(adjective):
        """ Helper function for checking whether the specified adjective is a comparative or superlative """
        # Count non-consecutive vowels
        vowel_seq = list()
        for ch in adjective:
            if ch in VOWELS:
                if len(vowel_seq) == 0:
                    vowel_seq.append(1)
                else:
                    if vowel_seq[-1] != 1:
                        vowel_seq.append(1)
            else:
                vowel_seq.append(0)
        if sum(vowel_seq) == 2:
            return True

    # Track samples to reduce duplicates
    orig_seen_samples = seen_samples
    # Process source sequence
    spacy_sent_rep = seed_parses[src_sent][0]
    spacy_tokens_lower = seed_parses[src_sent][1]
    ws_tokens = seed_parses[src_sent][2]
    spacy_to_ws_map = seed_parses[src_sent][4]

    spacy_src_rep_ids = list()
    spacy_rel_pos_ids = list()
    adversarial_samples = list()
    tokens_to_modify = list()

    # Only replace adjectives if they modify a noun to reduce ungrammatical samples
    for rep_id, rep in enumerate(spacy_sent_rep):
        if rep.pos_ in ['NOUN'] and rep_id != src_rep_loc:
            # Get rep lemma
            rep_lemma = rep.lower_ if \
                rep.lemma_ == '-PRON-' or rep.lemma_.isdigit() else rep.lemma_.lower()
            rep_lemma = rep_lemma.strip(punctuation_plus_space)
            #  Check children for adjectives to replace
            children = [child for child in rep.children]
            for child in children:
                if child.pos_ == 'ADJ' and child.text not in [tpl[0] for tpl in tokens_to_modify] \
                        and child.text.lower().strip(string.punctuation) not in QUANTIFIERS:
                    if child.text[0] == child.text[0].lower():  # Exclude 'proper noun adjectives', e.g. 'the Spanish'
                        tokens_to_modify.append((child.text, child.i, rep.text, rep_lemma))

    # Check if attractor is permissible
    for adj, adj_loc, noun_token, noun_lemma in tokens_to_modify:
        if not disable_modifiers:
            modifier_tokens = general_modifier_tokens.get(noun_lemma, None)
            modifier_lemmas = general_modifier_lemmas.get(noun_lemma, None)
            if modifier_tokens is None:
                continue
            else:
                # Check whether the modified noun should be modified by the current attractor
                keep_attractor_term = _score_attractor_with_modifiers(attractor_term,
                                                                      attractor_table,
                                                                      modifier_tokens,
                                                                      modifier_lemmas,
                                                                      seed_attractor_tokens,
                                                                      adversarial_attractor_tokens)
                if not keep_attractor_term:
                    continue

        if adj_loc not in spacy_rel_pos_ids:

            if not disable_ngrams:
                # Avoid breaking-up collocations
                term_to_replace = adj.lower().strip(punctuation_plus_space)
                if filter_bigrams.get(term_to_replace, None) is not None and attractor_term not in INTENSIFIERS:
                    modified_term = noun_token.lower().strip(punctuation_plus_space)
                    bigram_count = filter_bigrams[term_to_replace].get(modified_term, 0)
                    if bigram_count >= 300:
                        continue
                # Filter with bigrams
                if filter_bigrams.get(attractor_term, None) is not None and attractor_term not in INTENSIFIERS:
                    modified_term = noun_token.lower().strip(punctuation_plus_space)
                    bigram_count = filter_bigrams[attractor_term].get(modified_term, 0.)
                    if bigram_count < 10:
                        if attractor_term.endswith('er'):
                            if filter_bigrams.get(attractor_term[:-2], None) is not None:
                                bigram_count = filter_bigrams[attractor_term[:-2]].get(modified_term, 0.)
                        if attractor_term.endswith('est'):
                            if filter_bigrams.get(attractor_term[:-3], None) is not None:
                                bigram_count = filter_bigrams[attractor_term[:-3]].get(modified_term, 0.)
                        if bigram_count < 10:
                            continue

            # Check if insertion constraints are violated
            if spacy_sent_rep[adj_loc].text.lower().strip(punctuation_plus_space) == attractor_term:
                continue
            try:
                left_context = spacy_sent_rep[adj_loc - 1]
            except IndexError:
                left_context = None
            try:
                right_context = spacy_sent_rep[adj_loc + 1]
            except IndexError:
                right_context = None
            if right_context is not None:
                if right_context.pos_ not in ['NOUN', 'PROPN'] or \
                        (right_context.text.lower().strip(punctuation_plus_space) == attractor_term):
                    continue
            if left_context is not None:
                if left_context.pos_ in ['ADJ', 'PROPN'] or left_context.text == '@-@':
                    continue
                if adj_loc > 1:
                    if spacy_sent_rep[adj_loc - 2].text in ['a', 'an', 'the']:
                        continue
            if adj_loc < (len(spacy_sent_rep) - 2):
                # Avoid modifying compounds (e.g. 'arm strength')
                if spacy_sent_rep[adj_loc + 2].pos_ in ['NOUN', 'PROPN']:
                    continue
            spacy_rel_pos_ids.append(adj_loc)

    # Detect appropriate positions
    for token_id, token in enumerate(spacy_tokens_lower):
        if token in BLACKLIST:
            continue
        # Remove punctuation, separate compounds
        sub_token_list = re.sub(r' +', ' ', token.translate(pct_stripper)).split()
        sub_token_list = [sub_token.strip(punctuation_plus_space) for sub_token in sub_token_list]
        for sub_token in sub_token_list:
            if sub_token == src_rep:
                spacy_src_rep_ids.append(token_id)
                break  # only one sub-token hit per token allowed

    if len(spacy_src_rep_ids) == 0 or len(spacy_rel_pos_ids) == 0:
        return adversarial_samples, orig_seen_samples
    else:
        attractor_len = len(attractor_term.split())
        # Restrict set of modified terms to a window around each occurrence of the ambiguous term
        if len(spacy_rel_pos_ids) > window_size > 0:
            truncated_spacy_rel_pos_ids = list()
            truncated_spacy_rel_pos_ids += sorted(spacy_rel_pos_ids, key=lambda x: abs(x - src_rep_loc))[:window_size]
            spacy_rel_pos_ids = list(set(truncated_spacy_rel_pos_ids))

        for token_id in spacy_rel_pos_ids:
            # Convert to whitespace token position
            ws_token_id = spacy_to_ws_map[token_id][0]

            # Account for a / an
            if ws_token_id > 0:
                if ws_tokens[ws_token_id - 1] == 'a':
                    for vowel in list(VOWELS):
                        if attractor_term.startswith(vowel):
                            ws_tokens[ws_token_id - 1] = 'an'
                if ws_tokens[ws_token_id - 1] == 'an':
                    for consonant in list(CONSONANTS):
                        if attractor_term.startswith(consonant):
                            ws_tokens[ws_token_id - 1] = 'a'

            # Replace (most) adjectives with similar adjective forms
            if attractor_term.endswith('er') and _is_non_positive(attractor_term):
                if not (spacy_tokens_lower[token_id].endswith('er') and _is_non_positive(spacy_tokens_lower[token_id])):
                    continue
            if attractor_term.endswith('est') and _is_non_positive(attractor_term):
                if not (spacy_tokens_lower[token_id].endswith('est') and
                        _is_non_positive(spacy_tokens_lower[token_id])):
                    continue
            if (not (attractor_term.endswith('er') or attractor_term.endswith('est'))) or \
                    (not _is_non_positive(attractor_term)):
                if (spacy_tokens_lower[token_id].endswith('er') or spacy_tokens_lower[token_id].endswith('est')) and \
                        _is_non_positive(spacy_tokens_lower[token_id]):
                    continue

            # Account for superlatives and ordinals
            change_det = False
            for suffix in ['est'] + ORDINAL_SUFFIXES:
                if attractor_term.endswith(suffix):
                    change_det = True
                    break
            if change_det:
                if ws_token_id > 0:
                    if ws_tokens[ws_token_id - 1] in ['a', 'an']:
                        ws_tokens[ws_token_id - 1] = 'the'

            # Generate samples by inserting the attractor in the neighborhood of each token of the appropriate POS
            new_sent_tokens = ws_tokens[:ws_token_id] + [attractor_term] + ws_tokens[ws_token_id + 1:]
            new_sent = ' '.join(new_sent_tokens)
            attractor_ws_ids = [ws_token_id + attr_tok_id for attr_tok_id in range(len(attractor_term.split()))]
            updated_ambiguous_term_ws_ids = list()
            updated_ambiguous_focus_term_ws_id = spacy_to_ws_map[src_rep_loc][0]
            for rep_id in spacy_src_rep_ids:
                updated_rep_id = spacy_to_ws_map[rep_id][0]
                if updated_rep_id >= ws_token_id:
                    updated_rep_id = updated_rep_id - len(ws_tokens[ws_token_id].split()) + attractor_len
                    if rep_id == src_rep_loc:
                        updated_ambiguous_focus_term_ws_id = updated_rep_id
                updated_ambiguous_term_ws_ids.append(updated_rep_id)

            assert ws_tokens[spacy_to_ws_map[src_rep_loc][0]] == new_sent_tokens[updated_ambiguous_focus_term_ws_id], \
                'Mismatch between token at ambiguous token position in the original sentence \'{}\' | \'{}\' ' \
                'and generated sample \'{}\' | \'{}\''.format(src_sent.strip(), spacy_to_ws_map[src_rep_loc][0],
                                                              new_sent, updated_ambiguous_focus_term_ws_id)
            assert updated_ambiguous_focus_term_ws_id in updated_ambiguous_term_ws_ids, \
                'Term ID adjustment mismatch: Focus term ID: {}, ambiguous term IDs: {}' \
                .format(updated_ambiguous_focus_term_ws_id, updated_ambiguous_term_ws_ids)

            # Check if duplicate
            if seen_samples.get(new_sent, None):
                if seen_samples[new_sent] == (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id):
                    continue
            else:
                seen_samples[new_sent] = (src_rep, updated_ambiguous_focus_term_ws_id, attractor_cluster_id)
                adversarial_samples.append((new_sent,
                                            updated_ambiguous_term_ws_ids,
                                            updated_ambiguous_focus_term_ws_id,
                                            attractor_ws_ids))
    return adversarial_samples, seen_samples


def _parse_seed(seed_sentence,
                adversarial_cluster,
                src_word_loc,
                attractor,
                seed_parses):
    """ Helper function that parses the seed sentence and caches the results for greater efficiency """
    # Process source sequence
    if not seed_parses.get(seed_sentence, None):
        spacy_sent_rep, spacy_tokens_lower, _, _, ws_tokens, ws_tokens_lower, _, spacy_to_ws_map = \
            _process_strings(seed_sentence,
                             nlp,
                             get_lemmas=False,
                             get_pos=True,
                             remove_stopwords=False,
                             replace_stopwords=False,
                             get_maps=True)

        sentence_modifiers = list()
        src_term_rep = spacy_sent_rep[src_word_loc]
        src_term_lemma = src_term_rep.lower_ if \
            src_term_rep.lemma_ == '-PRON-' or src_term_rep.lemma_.isdigit() else src_term_rep.lemma_.lower()
        src_term_lemma = src_term_lemma.strip(punctuation_plus_space)
        # Identify modifiers
        children = [child for child in src_term_rep.children]
        for child in children:
            # Obtain lemmas
            child_lemma = \
                child.lower_ if child.lemma_ == '-PRON-' or child.lemma_.isdigit() else child.lemma_.lower()
            child_lemma = child_lemma.strip(punctuation_plus_space)
            # Filter by pos
            if child.pos_ in MODIFIERS_POS_SET and child_lemma != src_term_lemma \
                    and child.text not in CONTRACTIONS and len(child_lemma) > 1:
                sentence_modifiers.append(child_lemma)
        # Evaluate head
        head = src_term_rep.head
        head_lemma = head.lower_ if head.lemma_ == '-PRON-' or head.lemma_.isdigit() else head.lemma_.lower()
        head_lemma = head_lemma.strip(punctuation_plus_space)
        # Filter by pos
        if head.pos_ in MODIFIERS_POS_SET and head_lemma != src_term_lemma \
                and head.text not in CONTRACTIONS and len(head_lemma) > 1:
            sentence_modifiers.append(head_lemma)

        seed_parses[seed_sentence] = \
            (spacy_sent_rep, spacy_tokens_lower, ws_tokens, ws_tokens_lower, spacy_to_ws_map, sentence_modifiers,
             (src_word_loc, adversarial_cluster, attractor))
    return seed_parses


def _score_attractor_with_modifiers(attractor,
                                    attractor_table,
                                    modifier_tokens,
                                    modifier_lemmas,
                                    seed_attractor_tokens,
                                    adversarial_attractor_tokens,
                                    metric='[SORTED ATTRACTORS BY FREQ]'):
    """ Helper function that scores attractors according to their 'typicality' respective the relevant clusters """

    # Look up attractor lemma
    attractor_lemma = attractor_table['[CONTEXT TOKENS]'][attractor]['[LEMMA]']
    # Check if lemma is among modifier lemmas
    if modifier_lemmas.get(attractor_lemma, None) is None:
        return False
    else:
        # Exclude rare observations
        if modifier_lemmas[attractor_lemma]['[MODIFIERS WITH FREQ]'] < 1:
            return False
    return True


def _reformat_modifiers(modifiers_entry, seed_cluster):
    """ Re-formats modifier table entries for faster lookup of scores """
    # Reformat seed modifiers
    modifier_tokens = dict()
    modifier_lemmas = dict()
    metric_keys = [key for key in modifiers_entry[seed_cluster].keys() if key.startswith('[MODIFIERS WITH ')]

    if not modifiers_entry.get(seed_cluster, None):
        return modifier_tokens, modifier_lemmas
    # Iterate
    for mod_lemma in modifiers_entry[seed_cluster]['[MODIFIERS]'].keys():
        # Restrict to adjectives
        if 'amod' in modifiers_entry[seed_cluster]['[MODIFIERS]'][mod_lemma]['[DEP TAGS]'] and \
                'ADJ' in modifiers_entry[seed_cluster]['[MODIFIERS]'][mod_lemma]['[POS]']:

            modifier_lemmas[mod_lemma] = dict()
            for metric in metric_keys:
                modifier_lemmas[mod_lemma][metric] = modifiers_entry[seed_cluster][metric][mod_lemma]
            modifier_lemmas[mod_lemma]['[NUM TOKENS]'] = \
                len(modifiers_entry[seed_cluster]['[MODIFIERS]'][mod_lemma]['[TOKENS]'])

            for mod_token in modifiers_entry[seed_cluster]['[MODIFIERS]'][mod_lemma]['[TOKENS]']:
                if not modifier_tokens.get(mod_token):
                    modifier_tokens[mod_token] = dict()
                    modifier_tokens[mod_token]['[LEMMA]'] = mod_lemma
                    for metric in metric_keys:
                        modifier_tokens[mod_token][metric] = modifiers_entry[seed_cluster][metric][mod_lemma]
                    modifier_tokens[mod_token]['[LEMMA COUNTS]'] = \
                        modifiers_entry[seed_cluster]['[MODIFIERS]'][mod_lemma]['[COUNTS]']

    return modifier_tokens, modifier_lemmas


def _reformat_general_modifiers(modifiers_table):
    """ Re-formats general modifier table entries for faster lookup of scores """
    # Reformat seed modifiers
    modifier_tokens = dict()
    modifier_lemmas = dict()

    # Iterate
    logging.info('Re-formatting general modifiers ...')
    for term_lemma in modifiers_table.keys():
        metric_keys = [key for key in modifiers_table[term_lemma].keys() if key.startswith('[MODIFIERS WITH ')]
        for mod_lemma in modifiers_table[term_lemma]['[MODIFIERS]'].keys():
            # Restrict to adjectives
            if 'amod' in modifiers_table[term_lemma]['[MODIFIERS]'][mod_lemma]['[DEP TAGS]'] and \
                    'ADJ' in modifiers_table[term_lemma]['[MODIFIERS]'][mod_lemma]['[POS]']:

                if modifier_lemmas.get(term_lemma, None) is None:
                    modifier_lemmas[term_lemma] = dict()
                modifier_lemmas[term_lemma][mod_lemma] = dict()
                for metric in metric_keys:
                    modifier_lemmas[term_lemma][mod_lemma][metric] = modifiers_table[term_lemma][metric][mod_lemma]
                modifier_lemmas[term_lemma][mod_lemma]['[NUM TOKENS]'] = \
                    len(modifiers_table[term_lemma]['[MODIFIERS]'][mod_lemma]['[TOKENS]'])

                for mod_token in modifiers_table[term_lemma]['[MODIFIERS]'][mod_lemma]['[TOKENS]']:
                    if modifier_tokens.get(term_lemma, None) is None:
                        modifier_tokens[term_lemma] = dict()
                    if not modifier_tokens[term_lemma].get(mod_token):
                        modifier_tokens[term_lemma][mod_token] = dict()
                        modifier_tokens[term_lemma][mod_token]['[LEMMA]'] = mod_lemma
                        for metric in metric_keys:
                            modifier_tokens[term_lemma][mod_token][metric] = \
                                modifiers_table[term_lemma][metric][mod_lemma]
                        modifier_tokens[term_lemma][mod_token]['[LEMMA COUNTS]'] = \
                            modifiers_table[term_lemma]['[MODIFIERS]'][mod_lemma]['[COUNTS]']

    return modifier_tokens, modifier_lemmas


def _reformat_attractors(attractors_entry, seed_cluster, adversarial_cluster):
    """ Re-formats modifier table entries for faster lookup of scores """
    # Reformat seed attractors
    seed_attractor_tokens = dict()
    adv_attractor_tokens = dict()
    # Reformat
    for cluster, table in [(seed_cluster, seed_attractor_tokens),  (adversarial_cluster, adv_attractor_tokens)]:
        metric_keys = [key for key in attractors_entry[cluster].keys() if key.startswith('[SORTED ')]
        for metric in metric_keys:
            for attractor_token, score in attractors_entry[cluster][metric]:
                if attractor_token not in table.keys():
                    table[attractor_token] = dict()
                table[attractor_token][metric] = score
    # Compute means
    for table, means in [(seed_attractor_tokens, dict()), (adv_attractor_tokens, dict())]:
        for metric in metric_keys:
            means[metric] = np.mean([scores[metric] for scores in list(table.values())])
        table['[MEANS]'] = means
    return seed_attractor_tokens, adv_attractor_tokens


def _inject_attractor_terms(seed_table_entry,
                            attractor_term_table_entry,
                            sense_clusters_entry,
                            modifier_table_entry,
                            general_modifier_tokens,
                            general_modifier_lemmas,
                            filter_bigrams,
                            adversarial_table_entry,
                            attractor_top_n,
                            max_seed_sentences,
                            pos_list,
                            generative_function,
                            generation_window_size,
                            seed_parses,
                            disable_modifiers,
                            disable_ngrams):

    """ Helper function for constructing adversarial samples via the insertion of attractor words. """

    def _get_term_pos_tag(_tag_list):
        """ Helper function for determining the 'canonical' POS tag on an attractor term. """
        # Sort known tags
        tag_map = dict()
        for _tag in _tag_list:
            tag_count = _tag_list.count(_tag)
            if not tag_map.get(tag_count, None):
                tag_map[tag_count] = [_tag]
            else:
                tag_map[tag_count].append(_tag)
        max_count = sorted(list(tag_map.keys()), reverse=True)[0]
        tag_set = tag_map[max_count]
        # Resolve potential ambiguity
        if len(tag_set) > 1:
            for _tag in tag_set:
                if _tag in pos_list:
                    return _tag
        return tag_set[0]

    # Track generated samples to avoid duplicates
    seen_samples = dict()
    for seed_cluster_id in seed_table_entry.keys():
        # Skip seed clusters for which no attractors are known
        if seed_cluster_id not in attractor_term_table_entry.keys():
            continue

        # Iterate over attractors in other sense clusters
        for attractor_term_cluster_id in attractor_term_table_entry.keys():
            # Ignore same-sense attractors during iteration
            if attractor_term_cluster_id == seed_cluster_id:
                continue

            # Reformat attractor entries for easier access
            seed_attractor_tokens, adv_attractor_tokens = \
                _reformat_attractors(attractor_term_table_entry, seed_cluster_id, attractor_term_cluster_id)

            seed_modifier_tokens, seed_modifier_lemmas = None, None
            # Reformat table entries for faster lookup
            if generative_function in ['insert_at_homograph', 'replace_at_homograph']:
                seed_modifier_tokens, seed_modifier_lemmas = \
                    _reformat_modifiers(modifier_table_entry, seed_cluster_id)
                if len(seed_modifier_tokens.keys()) == 0 or len(seed_modifier_lemmas.keys()) == 0:
                    continue

            # Order seed sentences by source sentence length in descending order
            # Sentence length is derived from the length of the respective POS sequence
            cluster_seed_tuples = sorted(seed_table_entry[seed_cluster_id], reverse=True, key=lambda x: len(x[2]))
            sorted_attractors = attractor_term_table_entry[attractor_term_cluster_id]['[SORTED ATTRACTORS BY FREQ]']

            # Look-up target senses associated with the used clusters
            seed_cluster_senses = sense_clusters_entry[seed_cluster_id]['[SENSES]']
            attractor_cluster_senses = sense_clusters_entry[attractor_term_cluster_id]['[SENSES]']

            # Iterate over top attractors
            used_attractor_terms = list()
            for attractor_term, attractor_term_score in sorted_attractors:

                # Terminate early
                if len(used_attractor_terms) > attractor_top_n > 0:
                    break

                # Strip punctuation
                attractor_term = attractor_term.strip(punctuation_plus_space)

                # Exclude quantifiers as attractors
                if attractor_term in QUANTIFIERS:
                    continue

                # Check if attractor has the correct POS tag
                if pos_list:
                    attractor_term_pos_list = attractor_term_table_entry[attractor_term_cluster_id][
                        '[CONTEXT TOKENS]'][attractor_term]['[POS TAGS]']
                    # Attractor's most frequent POS is chosen as canonical
                    attractor_term_pos = _get_term_pos_tag(attractor_term_pos_list)
                    if attractor_term_pos not in pos_list:
                        continue

                if generative_function in ['insert_at_homograph', 'replace_at_homograph'] and not disable_modifiers:
                    # Evaluate attractor term for importance
                    keep_attractor_term = \
                        _score_attractor_with_modifiers(attractor_term,
                                                        attractor_term_table_entry[attractor_term_cluster_id],
                                                        seed_modifier_tokens, seed_modifier_lemmas,
                                                        seed_attractor_tokens, adv_attractor_tokens)
                    if not keep_attractor_term:
                        continue

                # Perturb seed sentences
                for sent_tpl_id, sent_tpl in enumerate(cluster_seed_tuples):

                    # Terminate early
                    if sent_tpl_id >= max_seed_sentences > 0:
                        break

                    # Unpack
                    src_sent, tgt_sent, src_sent_pos, tgt_sent_pos, src_tgt_alignment, src_token, tgt_sense, \
                        seed_src_rep_loc, seed_ws_src_rep_loc, seed_tgt_sense_loc, seed_ws_tgt_sns_loc = sent_tpl

                    # Skip duplicate seeds
                    if seed_parses.get(src_sent, None):
                        if seed_parses[src_sent][-1] == (seed_src_rep_loc, attractor_term_cluster_id, attractor_term):
                            continue

                    # Parse seed source sentence
                    seed_parses = \
                        _parse_seed(src_sent, attractor_term_cluster_id, seed_src_rep_loc, attractor_term, seed_parses)

                    # Generate samples
                    if generative_function == 'insert_at_other':
                        adversarial_samples, seen_samples = \
                            _insert_attractor_at_other_nouns(src_sent, src_token, seed_src_rep_loc, attractor_term,
                                                             attractor_term_table_entry[attractor_term_cluster_id],
                                                             seed_attractor_tokens, adv_attractor_tokens,
                                                             general_modifier_tokens, general_modifier_lemmas,
                                                             filter_bigrams, generation_window_size, seen_samples,
                                                             attractor_term_cluster_id, seed_parses, disable_modifiers,
                                                             disable_ngrams)

                    elif generative_function == 'replace_at_homograph':
                        adversarial_samples, seen_samples = \
                            _replace_attractor_at_homograph(src_sent, src_token, seed_src_rep_loc, attractor_term,
                                                            attractor_term_cluster_id, filter_bigrams, seen_samples,
                                                            seed_parses, disable_ngrams)

                    elif generative_function == 'replace_at_other':
                        adversarial_samples, seen_samples = \
                            _replace_attractor_at_other_nouns(src_sent, src_token, seed_src_rep_loc, attractor_term,
                                                              attractor_term_table_entry[attractor_term_cluster_id],
                                                              seed_attractor_tokens, adv_attractor_tokens,
                                                              general_modifier_tokens, general_modifier_lemmas,
                                                              filter_bigrams, generation_window_size, seen_samples,
                                                              attractor_term_cluster_id, seed_parses, disable_modifiers,
                                                              disable_ngrams)

                    else:
                        adversarial_samples, seen_samples = \
                            _insert_attractor_at_homograph(src_sent, src_token, seed_src_rep_loc, attractor_term,
                                                           attractor_term_cluster_id, filter_bigrams, seen_samples,
                                                           seed_parses, disable_ngrams)

                    if len(adversarial_samples) > 0:
                        # Add entries to the adversarial samples table
                        if adversarial_table_entry.get(seed_cluster_id, None) is None:
                            adversarial_table_entry[seed_cluster_id] = dict()
                        if adversarial_table_entry[seed_cluster_id].get(attractor_term_cluster_id, None) is None:
                            adversarial_table_entry[seed_cluster_id][attractor_term_cluster_id] = list()

                        # Carry-over attractor scores
                        attractor_term_scores = adv_attractor_tokens[attractor_term]
                        # Expand sample table
                        for sample_tpl in adversarial_samples:
                            sample, all_src_token_ws_ids, focus_src_token_ws_id, attractor_ws_ids = sample_tpl
                            adversarial_entry = [sample, src_sent, tgt_sent,
                                                 attractor_term, attractor_term_scores,
                                                 focus_src_token_ws_id, all_src_token_ws_ids,
                                                 seed_src_rep_loc, seed_ws_src_rep_loc,
                                                 seed_tgt_sense_loc, seed_ws_tgt_sns_loc,
                                                 attractor_ws_ids,
                                                 seed_cluster_id, attractor_term_cluster_id,
                                                 list(seed_cluster_senses), list(attractor_cluster_senses)]
                            adversarial_table_entry[seed_cluster_id][attractor_term_cluster_id] \
                                .append(adversarial_entry)

                        # Update phrase trackers
                        if attractor_term not in used_attractor_terms:
                            used_attractor_terms.append(attractor_term)

                        # Report occasionally
                        if len(seen_samples.keys()) > 0 and len(seen_samples.keys()) % 10000 == 0:
                            logging.info(
                                'Generated {:d} adversarial samples for the current source term / form entry'
                                .format(len(seen_samples.keys())))

            try:
                # Deduplicate within the same attractor cluster
                unique_samples = dict()
                unique_entries = list()
                for entry in adversarial_table_entry[seed_cluster_id][attractor_term_cluster_id]:
                    if unique_samples.get(entry[0], None) is None:
                        unique_samples[entry[0]] = True
                        unique_entries.append(entry)
                adversarial_table_entry[seed_cluster_id][attractor_term_cluster_id] = unique_entries
            except KeyError:
                continue


def generate_from_words(attractors_path,
                        seed_sentences_path,
                        sense_clusters_path,
                        modifiers_path,
                        general_modifiers_path,
                        filter_bigrams_path,
                        out_file_path,
                        pos_list,
                        attractor_top_n,
                        max_seed_sentences,
                        generative_function,
                        generation_window_size,
                        disable_modifiers,
                        disable_ngrams):

    """ Generates adversarial samples by injecting attractor tokens into mined seed set sentences. """

    def _extraction_sites_to_seeds():
        """ Helper for converting sentences from which attractor terms had been extracted into seed sentences. """
        converted_table = dict()
        for attr_term in attractors_table.keys():
            converted_table[attr_term] = dict()
            if has_forms:
                for attr_form in attractors_table[attr_term].keys():
                    converted_table[attr_term][attr_form] = dict()
                    for attr_cluster in attractors_table[attr_term][attr_form].keys():
                        converted_table[attr_term][attr_form][attr_cluster] = \
                            attractors_table[attr_term][attr_form][attr_cluster]['[SENTENCE PAIRS]']
                        # Add source term locations
                        for sent_tpl_id, sent_tpl in enumerate(converted_table[attr_term][attr_form][attr_cluster]):
                            extended_sent_tpl = \
                                tuple([item for item in sent_tpl] + attractors_table[
                                    attr_term][attr_cluster]['[SOURCE TERM LOCATIONS]'][sent_tpl_id] +
                                      attractors_table[attr_term][attr_cluster]['[TARGET TERM LOCATIONS]'][sent_tpl_id])
                            converted_table[attr_term][attr_form][attr_cluster][sent_tpl_id] = extended_sent_tpl
            else:
                for attr_cluster in attractors_table[attr_term].keys():
                    converted_table[attr_term][attr_cluster] = \
                        attractors_table[attr_term][attr_cluster]['[SENTENCE PAIRS]']
                    # Add source term locations
                    for sent_tpl_id, sent_tpl in enumerate(converted_table[attr_term][attr_cluster]):
                        extended_sent_tpl = \
                            tuple([item for item in sent_tpl] +
                                  attractors_table[attr_term][attr_cluster]['[SOURCE TERM LOCATIONS]'][sent_tpl_id] +
                                  attractors_table[attr_term][attr_cluster]['[TARGET TERM LOCATIONS]'][sent_tpl_id])
                        converted_table[attr_term][attr_cluster][sent_tpl_id] = extended_sent_tpl
        return converted_table

    def _show_stats():
        """ Helper for reporting on the generation process. """
        # Estimate total number of generated adversarial samples
        true_clusters_covered = 0
        adv_clusters_covered = 0
        nums_adv_samples = list()
        for src_term2 in adversarial_table.keys():
            true_clusters = list()
            adv_clusters = list()
            if has_forms:
                for src_form2 in adversarial_table[src_term2].keys():
                    true_clusters += adversarial_table[src_term2][src_form2].keys()
                    for true_cluster in adversarial_table[src_term2][src_form2].keys():
                        adv_clusters += adversarial_table[src_term2][src_form2][true_cluster].keys()
                        for adv_cluster in adversarial_table[src_term2][src_form2][true_cluster].keys():
                            num_adv_samples = len(adversarial_table[src_term2][src_form2][true_cluster][adv_cluster])
                            nums_adv_samples.append(num_adv_samples)
            else:
                true_clusters = adversarial_table[src_term2].keys()
                for true_cluster in adversarial_table[src_term2].keys():
                    adv_clusters += adversarial_table[src_term2][true_cluster].keys()
                    for adv_cluster in adversarial_table[src_term2][true_cluster].keys():
                        num_adv_samples = len(adversarial_table[src_term2][true_cluster][adv_cluster])
                        nums_adv_samples.append(num_adv_samples)

            true_clusters_covered += len(list(set(true_clusters)))
            adv_clusters_covered += len(list(set(adv_clusters)))

        # Report (total num samples, mean + std per term, mean + std per cluster)
        logging.info('Terms processed: {:d}'.format(len(adversarial_table.keys())))
        logging.info('Adversarial samples generated in total: {:d}'.format(sum(nums_adv_samples)))

        logging.info('-' * 20)
        logging.info('Number of true sense clusters covered: {:d}'.format(true_clusters_covered))
        logging.info('Number of true sense clusters NOT covered: {:d}'
                     .format(num_true_clusters - true_clusters_covered))

        logging.info('-' * 20)
        logging.info('Number of adversarial sense clusters covered: {:d}'.format(adv_clusters_covered))
        logging.info('Number of adversarial sense clusters NOT covered: {:d}'
                     .format(num_true_clusters - adv_clusters_covered))

        logging.info('-' * 20)
        if len(nums_adv_samples) > 0:
            logging.info('Samples per cluster avg.: {:.4f} | Samples per cluster std.: {:.4f}'
                         .format(float(np.mean(nums_adv_samples)), float(np.std(nums_adv_samples))))
        else:
            logging.info('Samples per cluster avg.: 0 | Samples per cluster std.: 0')

    # Read-in attractor table
    logging.info('Reading-in attractor table ...')
    with open(attractors_path, 'r', encoding='utf8') as ap:
        attractors_table = json.load(ap)

    # Read-in seed sentence pairs
    if seed_sentences_path is not None:
        logging.info('Reading-in collected seed sentence pairs ...')
        with open(seed_sentences_path, 'r', encoding='utf8') as chp:
            seed_sentences = json.load(chp)
    else:
        logging.info('Re-using attractor extraction sites as seed sentences ...')
        seed_sentences = _extraction_sites_to_seeds()

    # Read-in known BabelNet cluster senses
    logging.info('Reading-in target sense tables ...')
    with open(sense_clusters_path, 'r', encoding='utf8') as scp:
        sense_clusters = json.load(scp)

    # Read-in modifier table
    logging.info('Reading-in modifier table ...')
    with open(modifiers_path, 'r', encoding='utf8') as mp:
        modifiers_table = json.load(mp)

    # Read-in general modifier table
    general_modifier_tokens = None
    general_modifier_lemmas = None
    logging.info('Reading-in general modifier table ...')
    with open(general_modifiers_path, 'r', encoding='utf8') as gmp:
        general_modifiers_table = json.load(gmp)
    if generative_function not in ['insert_at_homograph', 'replace_at_homograph']:
        general_modifier_tokens, general_modifier_lemmas = _reformat_general_modifiers(general_modifiers_table)

    # Read-in Wiki bigrams table
    logging.info('Reading-in Wikipedia bigrams ...')
    with open(filter_bigrams_path, 'r', encoding='utf8') as wbp:
        filter_bigrams_table = json.load(wbp)

    # Check data type
    test_key1 = list(attractors_table.keys())[0]
    test_key2 = list(attractors_table[test_key1].keys())[0]
    has_forms = '[SENTENCE PAIRS]' not in attractors_table[test_key1][test_key2].keys()

    # Estimate the number of sense clusters in the attractor phrase table
    num_true_clusters = 0
    for term in attractors_table.keys():
        if has_forms:
            term_clusters = list()
            for form in attractors_table[term].keys():
                term_clusters += list(attractors_table[term][form].keys())
            num_true_clusters += len(list(set(term_clusters)))
        else:
            num_true_clusters += len(attractors_table[term].keys())

    # Initialize adversarial sample table
    adversarial_table = dict()

    # Cache seed sentence parses to avoid re-computation
    seed_parses = dict()

    # Iterate over seed sentence pairs
    for term_id, term in enumerate(seed_sentences.keys()):
        logging.info('Looking-up the term \'{:s}\''.format(term))
        if has_forms:
            for form in seed_sentences[term].keys():
                # Look up corresponding attractor table entry
                try:
                    attractor_table_entry = attractors_table[term][form]
                except KeyError:
                    continue

                # Look up corresponding modifier table entry
                try:
                    modifier_table_entry = modifiers_table[term]
                except KeyError:
                    modifier_table_entry = None

                # Create a new entry for the adversarial table
                if not adversarial_table.get(term, None):
                    adversarial_table[term] = dict()
                if not adversarial_table[term].get(form, None):
                    adversarial_table[term][form] = dict()

                # Generate samples
                _inject_attractor_terms(seed_sentences[term][form],
                                        attractor_table_entry,
                                        sense_clusters[term],
                                        modifier_table_entry,
                                        general_modifier_tokens,
                                        general_modifier_lemmas,
                                        filter_bigrams_table,
                                        adversarial_table[term][form],
                                        attractor_top_n,
                                        max_seed_sentences,
                                        pos_list,
                                        generative_function,
                                        generation_window_size,
                                        seed_parses,
                                        disable_modifiers,
                                        disable_ngrams)

        else:
            # Look up corresponding attractor table entry
            try:
                attractor_table_entry = attractors_table[term]
            except KeyError:
                continue

            # Look up corresponding modifier table entry
            try:
                modifier_table_entry = modifiers_table[term]
            except KeyError:
                modifier_table_entry = None

            # Create a new entry for the adversarial table
            if not adversarial_table.get(term, None):
                adversarial_table[term] = dict()

            # Generate samples
            _inject_attractor_terms(seed_sentences[term],
                                    attractor_table_entry,
                                    sense_clusters[term],
                                    modifier_table_entry,
                                    general_modifier_tokens,
                                    general_modifier_lemmas,
                                    filter_bigrams_table,
                                    adversarial_table[term],
                                    attractor_top_n,
                                    max_seed_sentences,
                                    pos_list,
                                    generative_function,
                                    generation_window_size,
                                    seed_parses,
                                    disable_modifiers,
                                    disable_ngrams)

        if term_id > 0 and term_id % 10 == 0:
            # Display stats
            _show_stats()

    # Display stats
    logging.info('FINAL REPORT')
    _show_stats()

    # Save adversarial samples to disc
    file_type = '_'.join(attractors_path[:-5].split('/')[-1].split('_')[2:])
    if generative_function == 'insert_at_homograph':
        adversarial_samples_path = '{:s}_{:s}_words_{:s}.json'\
            .format(out_file_path[:-5], file_type, generative_function)
    else:
        adversarial_samples_path = '{:s}_{:s}_words_{:s}_{:d}.json'\
            .format(out_file_path[:-5], file_type, generative_function, generation_window_size)
    # Report
    logging.info('Saving generated adversarial samples to {:s}'.format(adversarial_samples_path))
    with open(adversarial_samples_path, 'w', encoding='utf8') as wsp:
        json.dump(adversarial_table, wsp, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attractors_path', type=str, required=True,
                        help='path to the JSON file containing the extracted attractor terms')
    parser.add_argument('--seed_sentences_path', type=str, required=True,
                        help='path to the JSON file containing the extracted seed sentence pairs')
    parser.add_argument('--sense_clusters_path', type=str, default=None,
                        help='path to the JSON file containing scraped BabelNet sense clusters')
    parser.add_argument('--modifiers_path', type=str, required=True,
                        help='path to the JSON file containing the extracted modifier terms for the ambiguous nouns')
    parser.add_argument('--general_modifiers_path', type=str, required=True,
                        help='path to the JSON file containing the general extracted modifier terms')
    parser.add_argument('--filter_bigrams_path', type=str, required=True,
                        help='path to the JSON file containing Wikipedia bigrams, used to avoid breaking up '
                             'collocations during attractor insertion')
    parser.add_argument('--out_file_path', type=str, required=True,
                        help='path to which the seed sentence pairs should be saved')
    parser.add_argument('--filter_by_pos', action='store_true', help='only considers attractors of certain POS')
    parser.add_argument('--attractor_top_n', type=int, default=-1,
                        help='specifies the number of attractors for which attractor phrases are to be extracted')
    parser.add_argument('--max_seed_sentences', type=int, default=-1,
                        help='specifies the maximum number of seed sentences to be modified via attractor '
                             'insertion')
    parser.add_argument('--generative_function', type=str, choices=['insert_at_homograph', 'insert_at_other',
                                                                    'replace_at_homograph', 'replace_at_other'],
                        default='insert_single', help='picks the function used to add attractors to seed sentences')
    parser.add_argument('--generation_window_size', type=int, default=-1,
                        help='defines the size of the window considered for attractor insertion and replacement')
    parser.add_argument('--disable_modifiers', action='store_true',
                        help='disables filtering of attractors based on cross-cluster modifiers')
    parser.add_argument('--disable_ngrams', action='store_true',
                        help='disables filtering of attractors based on known source language n-grams')
    parser.add_argument('--src_lang', type=str, default=None,
                        help='denotes the language ID of the source sentences, e.g. \'en\'')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.out_file_path.split('/')[:-1])
    file_name = args.out_file_path.split('/')[-1]
    file_name = '.'.join(file_name.split('.')[:-1])
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
    spacy_map = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
    try:
        # nlp = spacy.load(spacy_map[args.src_lang], disable=['tagger', 'parser', 'ner', 'textcat'])
        nlp = spacy.load(spacy_map[args.src_lang], disable=['ner', 'textcat'])
    except KeyError:
        logging.info('SpaCy does not support the language {:s}. Exiting.'.format(args.src_lang))
        sys.exit(0)
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'
    # Import stopword list
    if args.src_lang == 'en':
        STOP_WORDS = stopwords.words('english')
    else:
        STOP_WORDS = []

    pos_filter_list = WORD_ATTRACTOR_TAGS if args.filter_by_pos else None

    generate_from_words(args.attractors_path,
                        args.seed_sentences_path,
                        args.sense_clusters_path,
                        args.modifiers_path,
                        args.general_modifiers_path,
                        args.filter_bigrams_path,
                        args.out_file_path,
                        pos_filter_list,
                        args.attractor_top_n,
                        args.max_seed_sentences,
                        args.generative_function,
                        args.generation_window_size,
                        args.disable_modifiers,
                        args.disable_ngrams)

