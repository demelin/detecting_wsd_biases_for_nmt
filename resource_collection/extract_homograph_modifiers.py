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


def _extract_cluster_modifiers(attractors_entry, seen_parses, pos_to_keep):

    """ Searches through all matched sentence pairs within a sense cluster, parses them, and compiles a list of
    relevant modifiers of the ambiguous term """

    # Store cluster-level information
    cluster_modifiers = dict()

    # Iterate over sentence pairs
    for sent_tpl_id, sent_tpl in enumerate(attractors_entry['[SENTENCE PAIRS]']):
        # Unpack
        src_line, _, _, _, _, _, _ = sent_tpl

        # Process source sequence
        if not seen_parses.get(src_line, None):
            spacy_sent_rep, _, _, _, _, _, _, _ = \
                _process_strings(src_line,
                                 nlp,
                                 get_lemmas=False,
                                 get_pos=False,
                                 remove_stopwords=False,
                                 replace_stopwords=False,
                                 get_maps=False)

            seen_parses[src_line] = spacy_sent_rep
        else:
            spacy_sent_rep = seen_parses[src_line]

        # Look up position of the ambiguous term
        spacy_loc, ws_loc = attractors_entry['[SOURCE TERM LOCATIONS]'][sent_tpl_id]
        src_term_rep = spacy_sent_rep[spacy_loc]
        src_term_lemma = src_term_rep.lower_ if \
            src_term_rep.lemma_ == '-PRON-' or src_term_rep.lemma_.isdigit() else src_term_rep.lemma_.lower()
        src_term_lemma = src_term_lemma.strip(punctuation_plus_space)

        # Store sentence-wise information
        sentence_modifiers = dict()

        # Ignore compounds
        if spacy_loc < len(spacy_sent_rep) - 1:
            if spacy_sent_rep[spacy_loc + 1].pos_ in COMPOUND_POS_LIST:
                continue

        # Lookup children
        children = [child for child in src_term_rep.children]
        for child in children:
            # Obtain lemmas
            child_lemma = \
                child.lower_ if child.lemma_ == '-PRON-' or child.lemma_.isdigit() else child.lemma_.lower()
            child_lemma = child_lemma.strip(punctuation_plus_space)
            child_token = child.text.lower().strip(punctuation_plus_space)
            # Filter by pos
            if child.pos_ in pos_to_keep and child_lemma != src_term_lemma and \
                    child_token not in CONTRACTIONS and len(child_lemma) > 1 and child.dep_ != 'compound':
                # Extend modifier table
                if not sentence_modifiers.get(child_lemma, None):
                    sentence_modifiers[child_lemma] = {'[TOKENS]': list(),
                                                       '[POS]': list(),
                                                       '[DEP TAGS]': list(),
                                                       '[ROLE]': list(),
                                                       '[COUNTS]': 0}
                sentence_modifiers[child_lemma]['[TOKENS]'].append(child_token)
                sentence_modifiers[child_lemma]['[POS]'].append(child.pos_)
                sentence_modifiers[child_lemma]['[DEP TAGS]'].append(child.dep_)
                sentence_modifiers[child_lemma]['[ROLE]'].append('child')
                sentence_modifiers[child_lemma]['[COUNTS]'] += 1
        # Evaluate head
        head = src_term_rep.head
        head_lemma = head.lower_ if head.lemma_ == '-PRON-' or head.lemma_.isdigit() else head.lemma_.lower()
        head_lemma = head_lemma.strip(punctuation_plus_space)
        head_token = head.text.lower().strip(punctuation_plus_space)
        # Filter by pos
        if head.pos_ in pos_to_keep and head_lemma != src_term_lemma and \
                head_token not in CONTRACTIONS and len(head_lemma) > 1:
            # Extend modifier table
            if not sentence_modifiers.get(head_lemma, None):
                sentence_modifiers[head_lemma] = {'[TOKENS]': list(),
                                                  '[POS]': list(),
                                                  '[DEP TAGS]': list(),
                                                  '[ROLE]': list(),
                                                  '[COUNTS]': 0}
            sentence_modifiers[head_lemma]['[TOKENS]'].append(head_token)
            sentence_modifiers[head_lemma]['[POS]'].append(head.pos_)
            sentence_modifiers[head_lemma]['[DEP TAGS]'].append(head.dep_)
            sentence_modifiers[head_lemma]['[ROLE]'].append('head')
            sentence_modifiers[head_lemma]['[COUNTS]'] += 1

        # Consolidate sentence-level information
        for lemma in sentence_modifiers:
            if not cluster_modifiers.get(lemma, None):
                cluster_modifiers[lemma] = sentence_modifiers[lemma]
            else:
                for key in cluster_modifiers[lemma].keys():
                    cluster_modifiers[lemma][key] += sentence_modifiers[lemma][key]

    if len(cluster_modifiers.keys()) == 0:
        return None
    else:
        return cluster_modifiers


def extract_all_modifiers(attractors_path):

    """ Extracts certain classes of words that modify ambiguous terms of interest from parallel sentence pairs. """

    def _score_modifiers(entry):
        """ Helper function for calculation various modifier relevance metrics at cluster level """
        # Compute modifier frequency for each cluster
        modifier_counts = dict()
        cluster_sizes = dict()
        modifier_frequencies = dict()
        for sense in entry.keys():
            # Compute modifier total
            cluster_sizes[sense] = sum([entry[sense][modifier]['[COUNTS]'] for modifier in entry[sense].keys()])
            for modifier in entry[sense].keys():
                if not modifier_frequencies.get(modifier, None):
                    modifier_counts[modifier] = dict()
                    modifier_frequencies[modifier] = dict()
                modifier_counts[modifier][sense] = entry[sense][modifier]['[COUNTS]']
                modifier_frequencies[modifier][sense] = entry[sense][modifier]['[COUNTS]'] / cluster_sizes[sense]
        # Add smoothing for PMI computation
        smoothed_modifier_counts = dict()
        smoothed_cluster_sizes = {sense: 0 for sense in entry.keys()}
        for modifier in modifier_counts.keys():
            smoothed_modifier_counts[modifier] = dict()
            for sense in entry.keys():
                sense_count = modifier_counts[modifier].get(sense, 0)
                smoothed_sense_count = sense_count + 100  # smoothing factor
                smoothed_modifier_counts[modifier][sense] = smoothed_sense_count
                smoothed_cluster_sizes[sense] += smoothed_sense_count
        # Compute smoothed counts total
        smoothed_modifier_total = 0
        for modifier in smoothed_modifier_counts:
            smoothed_modifier_total += sum(smoothed_modifier_counts[modifier].values())

        # Score
        updated_entry = dict()
        num_clusters_entry = len(entry.keys())
        for sense in entry.keys():
            modifiers_with_freq = list()
            modifiers_with_ratio_within = list()
            modifiers_with_ratio_across = list()
            modifiers_with_mim = list()
            modifiers_with_pmi = list()
            # Compute overall number of modifiers for this target sense
            for modifier in entry[sense].keys():
                # FREQ
                modifiers_with_freq.append((modifier, entry[sense][modifier]['[COUNTS]']))
                # RATIO WITHIN
                modifier_ratio_within = modifier_frequencies[modifier][sense]
                modifiers_with_ratio_within.append((modifier, modifier_ratio_within))
                # RATIO ACROSS
                total_modifiers_across_clusters = sum(modifier_counts[modifier].values())
                modifier_ratio_across = modifier_counts[modifier][sense] / total_modifiers_across_clusters
                modifiers_with_ratio_across.append((modifier, modifier_ratio_across / cluster_sizes[sense]))
                # Compute inverse local cluster frequency
                inverse_local_cluster_frequency = \
                    np.log(num_clusters_entry / sum(list(modifier_frequencies[modifier].values())))
                # MIM (Modifier Importance Metric)
                modifier_mim = modifier_ratio_within * inverse_local_cluster_frequency
                modifiers_with_mim.append((modifier, modifier_mim))
                # PMI
                joined_prob = smoothed_modifier_counts[modifier][sense] / smoothed_modifier_total
                marginal_prob_mod = sum(smoothed_modifier_counts[modifier].values()) / smoothed_modifier_total
                marginal_prob_sense = smoothed_cluster_sizes[sense] / smoothed_modifier_total
                modifier_sense_pmi = np.log2(joined_prob / (marginal_prob_mod * marginal_prob_sense))
                modifiers_with_pmi.append((modifier, modifier_sense_pmi))
            # Integrate
            updated_entry[sense] = dict()
            updated_entry[sense]['[MODIFIERS]'] = {k: v for k, v in entry[sense].items()}

            updated_entry[sense]['[MODIFIERS WITH FREQ]'] = dict()
            for mwf, score in modifiers_with_freq:
                updated_entry[sense]['[MODIFIERS WITH FREQ]'][mwf] = score

            updated_entry[sense]['[MODIFIERS WITH RATIO WITHIN]'] = dict()
            for mrw, score in modifiers_with_ratio_within:
                updated_entry[sense]['[MODIFIERS WITH RATIO WITHIN]'][mrw] = score

            updated_entry[sense]['[MODIFIERS WITH RATIO ACROSS]'] = dict()
            for mra, score in modifiers_with_ratio_across:
                updated_entry[sense]['[MODIFIERS WITH RATIO ACROSS]'][mra] = score

            updated_entry[sense]['[MODIFIERS WITH MIM]'] = dict()
            for mwm, score in modifiers_with_mim:
                updated_entry[sense]['[MODIFIERS WITH MIM]'][mwm] = score

            updated_entry[sense]['[MODIFIERS WITH PMI]'] = dict()
            for mwp, score in modifiers_with_pmi:
                updated_entry[sense]['[MODIFIERS WITH PMI]'][mwp] = score

        return updated_entry

    # Check file type
    is_nouns_only = '_nouns' in attractors_path

    # Read-in the attractors table
    logging.info('Reading-in known attractors ...')
    with open(attractors_path, 'r', encoding='utf8') as atp:
        attractors_table = json.load(atp)

    # Initialize the output table
    modifiers_table = dict()

    # Cache seed sentence parses to avoid re-computation
    seen_parses = dict()

    # Track counts
    modifiers_per_cluster = list()

    # Iterate
    for term_id, term in enumerate(attractors_table.keys()):
        logging.info('Looking-up the term \'{:s}\''.format(term))
        if is_nouns_only:
            running_modifier_counts = dict()
            for form in attractors_table[term].keys():
                for cluster in attractors_table[term][form].keys():
                    cluster_modifiers = _extract_cluster_modifiers(attractors_table[term][form][cluster],
                                                                   seen_parses,
                                                                   MODIFIERS_POS_SET)
                    if cluster_modifiers is not None:
                        # Extend modifiers table
                        if not modifiers_table.get(term, None):
                            modifiers_table[term] = dict()
                        if not modifiers_table[term].get(cluster, None):
                            modifiers_table[term][cluster] = cluster_modifiers
                            running_modifier_counts[cluster] = len(cluster_modifiers.keys())
                        else:
                            # Consolidate modifiers across forms
                            for mod in cluster_modifiers:
                                if mod not in modifiers_table[term][cluster].keys():
                                    modifiers_table[term][cluster][mod] = cluster_modifiers[mod]
                                else:
                                    for key in cluster_modifiers[mod].keys():
                                        modifiers_table[term][cluster][mod][key] += cluster_modifiers[mod][key]
                            running_modifier_counts[cluster] += len(cluster_modifiers.keys())
                        modifiers_per_cluster += list(running_modifier_counts.values())

        else:
            for cluster in attractors_table[term].keys():
                cluster_modifiers = _extract_cluster_modifiers(attractors_table[term][cluster],
                                                               seen_parses,
                                                               MODIFIERS_POS_SET)
                if cluster_modifiers is not None:
                    # Extend modifiers table
                    if not modifiers_table.get(term, None):
                        modifiers_table[term] = dict()
                    modifiers_table[term][cluster] = cluster_modifiers
                    modifiers_per_cluster.append(len(cluster_modifiers.keys()))

        # Report occasionally
        if term_id > 0 and term_id % 10 == 0:
            logging.info('Found a total of {:d} modifiers'.format(sum(modifiers_per_cluster)))
            logging.info('Modifiers per cluster avg.: {:.3f}, std.: {:.3f}'
                         .format(np.mean(modifiers_per_cluster), np.std(modifiers_per_cluster)))

    for term in modifiers_table.keys():
        modifiers_table[term] = _score_modifiers(modifiers_table[term])
    logging.info('Scoring of modifiers completed!')

    # Final report
    logging.info('FINAL REPORT')
    logging.info('Found a total of {:d} modifiers'.format(sum(modifiers_per_cluster)))
    logging.info('Modifiers per cluster avg.: {:.3f}, std.: {:.3f}'
                 .format(np.mean(modifiers_per_cluster), np.std(modifiers_per_cluster)))

    logging.info('Saving extracted modifiers ...')
    modifiers_out_file_path = attractors_path[:-5] + '_modifiers.json'
    with open(modifiers_out_file_path, 'w', encoding='utf8') as mof:
        json.dump(modifiers_table, mof, indent=3, sort_keys=True, ensure_ascii=False)
    logging.info('Saved the modifiers table to {:s}'.format(modifiers_out_file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attractors_path', type=str, required=True,
                        help='path to the JSON file containing the extracted attractor terms')
    parser.add_argument('--src_lang', type=str, default=None,
                        help='denotes the language ID of the source sentences, e.g. \'en\'')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.attractors_path.split('/')[:-1])
    file_name = args.attractors_path.split('/')[-1]
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

    extract_all_modifiers(args.attractors_path)

