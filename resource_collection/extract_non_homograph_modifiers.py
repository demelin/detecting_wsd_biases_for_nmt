import os
import re
import sys
import copy
import json
import spacy
import string
import logging
import argparse

import numpy as np
from nltk.corpus import stopwords

MODIFIERS_POS_SET = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP']
COMPOUND_POS_LIST = ['NOUN', 'PROPN']
# See:
# https://spacy.io/api/annotation#dependency-parsing
# https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
SRC_TAG_SET = ['agent', 'attr', 'dative', 'dobj', 'nsubj', 'nsubjpass', 'obj', 'oprd']
CONTRACTIONS = \
    ['t', 've', 'd', 'ye', 'e', 'er', 's', 'g', 'n', 'll', 're', 'm', 'a', 'o', 're', 'y', 'gon', 'wan', 'na']


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


def _extract_modifiers(line, pos_to_keep):

    """ Parses a single line taken from a text corpus, identifying nouns and elements modifying them /
    being modified by them """

    spacy_sent_rep, _, _, _, _, _, _, _ = \
        _process_strings(line,
                         nlp,
                         get_lemmas=False,
                         get_pos=False,
                         remove_stopwords=False,
                         replace_stopwords=False,
                         get_maps=False)

    sentence_modifiers = dict()
    noun_tokens = dict()

    # Find modifiers
    for rep_id, rep in enumerate(spacy_sent_rep):
        if rep.pos_ == 'NOUN':

            # Ignore compounds
            if rep_id < len(spacy_sent_rep) - 1:
                if spacy_sent_rep[rep_id + 1].pos_ in COMPOUND_POS_LIST:
                    continue

            noun_lemma = \
                rep.lower_ if rep.lemma_ == '-PRON-' or rep.lemma_.isdigit() else rep.lemma_.lower()
            noun_lemma = noun_lemma.strip(punctuation_plus_space)
            if len(noun_lemma) == 0 or noun_lemma in CONTRACTIONS:
                continue
            noun_modifiers = dict()
            if not noun_tokens.get(noun_lemma, None):
                noun_tokens[noun_lemma] = list()
            noun_tokens[noun_lemma].append(rep.text.lower())

            # Find modifiers
            children = [child for child in rep.children]
            for child in children:
                # Obtain lemmas
                child_lemma = \
                    child.lower_ if child.lemma_ == '-PRON-' or child.lemma_.isdigit() else child.lemma_.lower()
                child_lemma = child_lemma.strip(punctuation_plus_space)
                child_token = child.text.lower().strip(punctuation_plus_space)
                # Filter by pos
                if child.pos_ in pos_to_keep and child_lemma != noun_lemma and \
                        child_token not in CONTRACTIONS and len(child_lemma) > 1:
                    # Extend modifier table
                    if not noun_modifiers.get(child_lemma, None):
                        noun_modifiers[child_lemma] = {'[TOKENS]': list(),
                                                       '[POS]': list(),
                                                       '[DEP TAGS]': list(),
                                                       '[ROLE]': list(),
                                                       '[COUNTS]': 0}
                    noun_modifiers[child_lemma]['[TOKENS]'].append(child_token)
                    noun_modifiers[child_lemma]['[POS]'].append(child.pos_)
                    noun_modifiers[child_lemma]['[DEP TAGS]'].append(child.dep_)
                    noun_modifiers[child_lemma]['[ROLE]'].append('child')
                    noun_modifiers[child_lemma]['[COUNTS]'] += 1
            # Evaluate head
            head = rep.head
            head_lemma = head.lower_ if head.lemma_ == '-PRON-' or head.lemma_.isdigit() else head.lemma_.lower()
            head_lemma = head_lemma.strip(punctuation_plus_space)
            head_token = head.text.lower().strip(punctuation_plus_space)
            # Filter by pos
            if head.pos_ in pos_to_keep and head_lemma != noun_lemma and \
                    head_token not in CONTRACTIONS and len(head_lemma) > 1:
                if not noun_modifiers.get(head_lemma, None):
                    noun_modifiers[head_lemma] = {'[TOKENS]': list(),
                                                  '[POS]': list(),
                                                  '[DEP TAGS]': list(),
                                                  '[ROLE]': list(),
                                                  '[COUNTS]': 0}
                noun_modifiers[head_lemma]['[TOKENS]'].append(head_token)
                noun_modifiers[head_lemma]['[POS]'].append(head.pos_)
                noun_modifiers[head_lemma]['[DEP TAGS]'].append(head.dep_)
                noun_modifiers[head_lemma]['[ROLE]'].append('head')
                noun_modifiers[head_lemma]['[COUNTS]'] += 1

            # Consolidate sentence-level information
            if len(noun_modifiers.keys()) > 0:
                if sentence_modifiers.get(noun_lemma, None) is None:
                    sentence_modifiers[noun_lemma] = noun_modifiers
                else:
                    for mod_lemma in noun_modifiers.keys():
                        if sentence_modifiers[noun_lemma].get(mod_lemma, None) is None:
                            sentence_modifiers[noun_lemma][mod_lemma] = noun_modifiers[mod_lemma]
                        else:
                            for key in sentence_modifiers[noun_lemma][mod_lemma].keys():
                                sentence_modifiers[noun_lemma][mod_lemma][key] += noun_modifiers[mod_lemma][key]
    return sentence_modifiers, noun_tokens


def extract_all_modifiers(corpus_path, out_path):

    """ Extracts certain classes of words that modify ambiguous terms of interest from parallel sentence pairs. """

    def _score_modifiers(entry,
                         modifier_counts_global,
                         modifier_frequencies_global,
                         noun_entry_sizes_global,
                         nouns_total,
                         curr_noun):
        """ Helper function for calculation various modifier relevance metrics at cluster level """
        # Add smoothing for PMI computation
        smoothed_modifier_counts_global = dict()
        smoothed_noun_entry_sizes_global = {n: 0 for n in noun_entry_sizes_global.keys()}
        for mod in modifier_counts_global.keys():
            smoothed_modifier_counts_global[mod] = dict()
            for n in modifier_counts_global[mod].keys():
                n_count = modifier_counts_global[mod].get(n, 0)
                smoothed_n_count = n_count + 100  # smoothing factor
                smoothed_modifier_counts_global[mod][n] = smoothed_n_count
                smoothed_noun_entry_sizes_global[n] += smoothed_n_count
        # Compute smoothed counts total
        smoothed_modifier_total = 0
        for mod in smoothed_modifier_counts_global:
            smoothed_modifier_total += sum(smoothed_modifier_counts_global[mod].values())

        modifiers_with_freq = list()
        modifiers_with_ratio_within = list()
        modifiers_with_ratio_across = list()
        modifiers_with_mim = list()
        modifiers_with_pmi = list()
        for mod in entry.keys():
            # FREQ
            modifiers_with_freq.append((mod, entry[mod]['[COUNTS]']))
            # RATIO WITHIN
            modifier_ratio_within = modifier_frequencies_global[mod][curr_noun]
            modifiers_with_ratio_within.append((mod, modifier_ratio_within))
            # RATIO ACROSS
            total_modifiers_across_clusters = sum(modifier_counts_global[mod].values())
            modifier_ratio_across = modifier_counts_global[mod][curr_noun] / total_modifiers_across_clusters
            modifiers_with_ratio_across.append((mod, modifier_ratio_across / noun_entry_sizes_global[curr_noun]))
            # Compute inverse noun frequency
            inverse_noun_frequency = np.log(nouns_total / sum(list(modifier_frequencies_global[mod].values())))
            # MIM
            modifier_mim = modifier_frequencies_global[mod][curr_noun] * inverse_noun_frequency
            modifiers_with_mim.append((mod, modifier_mim))
            # PMI
            joined_prob = smoothed_modifier_counts_global[mod][curr_noun] / smoothed_modifier_total
            marginal_prob_mod = sum(smoothed_modifier_counts_global[modifier].values()) / smoothed_modifier_total
            marginal_prob_noun = smoothed_noun_entry_sizes_global[curr_noun] / smoothed_modifier_total
            modifier_sense_pmi = np.log2(joined_prob / (marginal_prob_mod * marginal_prob_noun))
            modifiers_with_pmi.append((modifier, modifier_sense_pmi))
        # Integrate
        updated_entry = dict()
        updated_entry['[MODIFIERS]'] = {m: v for m, v in entry.items()}

        updated_entry['[MODIFIERS WITH FREQ]'] = dict()
        for mwf, score in modifiers_with_freq:
            updated_entry['[MODIFIERS WITH FREQ]'][mwf] = score

        updated_entry['[MODIFIERS WITH RATIO WITHIN]'] = dict()
        for mrw, score in modifiers_with_ratio_within:
            updated_entry['[MODIFIERS WITH RATIO WITHIN]'][mrw] = score

        updated_entry['[MODIFIERS WITH RATIO ACROSS]'] = dict()
        for mra, score in modifiers_with_ratio_across:
            updated_entry['[MODIFIERS WITH RATIO ACROSS]'][mra] = score

        updated_entry['[MODIFIERS WITH MIM]'] = dict()
        for mwm, score in modifiers_with_mim:
            updated_entry['[MODIFIERS WITH MIM]'][mwm] = score

        updated_entry['[MODIFIERS WITH PMI]'] = dict()
        for mwp, score in modifiers_with_pmi:
            updated_entry['[MODIFIERS WITH PMI]'][mwp] = score
        return updated_entry

    # Initialize the output table
    modifiers_table = dict()
    noun_tokens_table = dict()

    # Track counts
    line_count = 0
    noun_count = 0
    mod_count = 0

    # Iterate
    with open(corpus_path, 'r', encoding='utf8') as out_file:
        for line_id, line in enumerate(out_file):
            line_modifiers, noun_tokens = _extract_modifiers(line.strip(), MODIFIERS_POS_SET)
            if len(line_modifiers.keys()) <= 0:
                continue
            # Extend modifiers table
            for noun in line_modifiers.keys():
                if modifiers_table.get(noun, None) is None:
                    modifiers_table[noun] = line_modifiers[noun]
                    noun_tokens_table[noun] = noun_tokens[noun]
                    noun_count += 1
                    mod_count += len(line_modifiers[noun].keys())
                else:
                    noun_tokens_table[noun] += noun_tokens[noun]
                    for modifier in line_modifiers[noun].keys():
                        if modifiers_table[noun].get(modifier, None) is None:
                            modifiers_table[noun][modifier] = line_modifiers[noun][modifier]
                            mod_count += 1
                        else:
                            for key in modifiers_table[noun][modifier].keys():
                                modifiers_table[noun][modifier][key] += line_modifiers[noun][modifier][key]
            line_count += 1

            # Report occasionally
            if line_count > 0 and line_count % 10000 == 0:
                logging.info('Parsed {:d} lines successfully'.format(line_id))
                logging.info('Found a total of {:d} modifiers for {:d} unique noun lemmas'
                             .format(mod_count, noun_count))

            # Checkpoint
            if line_count > 0 and line_count % 100000 == 0:
                checkpoint_table = copy.deepcopy(modifiers_table)
                modifier_frequencies = dict()
                modifier_counts = dict()
                noun_entry_sizes = dict()
                num_nouns = 0
                for noun in modifiers_table.keys():
                    num_nouns += 1
                    # Compute modifier total
                    noun_entry_sizes[noun] = \
                        sum([modifiers_table[noun][modifier]['[COUNTS]'] for modifier in modifiers_table[noun].keys()])
                    for modifier in modifiers_table[noun].keys():
                        if not modifier_frequencies.get(modifier, None):
                            modifier_counts[modifier] = dict()
                            modifier_frequencies[modifier] = dict()
                        modifier_counts[modifier][noun] = modifiers_table[noun][modifier]['[COUNTS]']
                        modifier_frequencies[modifier][noun] = \
                            modifiers_table[noun][modifier]['[COUNTS]'] / noun_entry_sizes[noun]

                for noun in modifiers_table.keys():
                    checkpoint_table[noun] = _score_modifiers(modifiers_table[noun], modifier_counts,
                                                              modifier_frequencies, noun_entry_sizes, num_nouns, noun)

                logging.info('Scoring of modifiers completed!')
                logging.info('Checkpointing extracted modifiers ...')
                modifiers_path = '{:s}.json'.format(out_path)
                tokens_path = '{:s}_noun_tokens.json'.format(out_path)
                with open(modifiers_path, 'w', encoding='utf8') as mof:
                    json.dump(checkpoint_table, mof, indent=3, sort_keys=True, ensure_ascii=False)
                logging.info('Saved the modifiers table to {:s}'.format(modifiers_path))
                with open(tokens_path, 'w', encoding='utf8') as tof:
                    json.dump(noun_tokens_table, tof, indent=3, sort_keys=True, ensure_ascii=False)
                logging.info('Saved noun tokens to {:s}'.format(tokens_path))


    # Score modifiers
    # Compute modifier frequency for each noun
    modifier_frequencies = dict()
    modifier_counts = dict()
    noun_entry_sizes = dict()
    num_nouns = 0
    for noun in modifiers_table.keys():
        num_nouns += 1
        # Compute modifier total
        noun_entry_sizes[noun] = \
            sum([modifiers_table[noun][modifier]['[COUNTS]'] for modifier in modifiers_table[noun].keys()])
        for modifier in modifiers_table[noun].keys():
            if not modifier_frequencies.get(modifier, None):
                modifier_counts[modifier] = dict()
                modifier_frequencies[modifier] = dict()
            modifier_counts[modifier][noun] = modifiers_table[noun][modifier]['[COUNTS]']
            modifier_frequencies[modifier][noun] = \
                modifiers_table[noun][modifier]['[COUNTS]'] / noun_entry_sizes[noun]

    for noun in modifiers_table.keys():
        modifiers_table[noun] = _score_modifiers(modifiers_table[noun], modifier_counts,
                                                 modifier_frequencies, noun_entry_sizes, num_nouns, noun)
    logging.info('Scoring of modifiers completed!')

    # Final report
    logging.info('FINAL REPORT')
    logging.info('Found a total of {:d} modifiers for {:d} unique noun lemmas'.format(mod_count, noun_count))

    logging.info('Saving extracted modifiers ...')
    modifiers_path = '{:s}.json'.format(out_path)
    tokens_path = '{:s}_noun_tokens.json'.format(out_path)
    with open(modifiers_path, 'w', encoding='utf8') as mof:
        json.dump(modifiers_table, mof, indent=3, sort_keys=True, ensure_ascii=False)
    logging.info('Saved the modifiers table to {:s}'.format(modifiers_path))
    with open(tokens_path, 'w', encoding='utf8') as tof:
        json.dump(noun_tokens_table, tof, indent=3, sort_keys=True, ensure_ascii=False)
    logging.info('Saved noun tokens to {:s}'.format(tokens_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True,
                        help='path to the text corpus')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to which the modifiers should be saved')
    parser.add_argument('--src_lang', type=str, default=None,
                        help='denotes the language ID of the source sentences, e.g. \'en\'')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.output_path.split('/')[:-1])
    file_name = args.output_path.split('/')[-1]
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

    extract_all_modifiers(args.corpus_path, args.output_path)

