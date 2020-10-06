import io
import os
import re
import sys
import json
import spacy
import torch
import pickle
import string
import logging
import argparse

import numpy as np
from nltk.corpus import wordnet as wn
from babelnetpy.babelnet import BabelNet


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



def _load_embeddings(file_path, preprocess):
    """ Helper function for reading-in pre-trained fasttext word embeddings;
    adopted from https://fasttext.cc/docs/en/english-vectors.html;
    pre-processing adopted from https://cs231n.github.io/neural-networks-2/;
    NOTE: Embeddings are used only for the clustering of target senses. """

    # Construct pickle paths
    token_path = file_path + '.tokens.json'
    base_path = file_path + '.base.pkl'
    centered_path = file_path + '.centered.pkl'
    reduced_path = file_path + '.reduced.pkl'
    whitened_path = file_path + '.whitened.pkl'
    pickle_path = base_path
    embedding_table = None
    embedding_dict = dict()
    loaded_from_file = False

    if not preprocess and os.path.isfile(base_path):
        with open(base_path, 'rb') as bpo:
            embedding_table = pickle.load(bpo)
        loaded_from_file = True
    elif preprocess == 'center' and os.path.isfile(centered_path):
        with open(centered_path, 'rb') as cpo:
            embedding_table = pickle.load(cpo)
        loaded_from_file = True
    elif preprocess == 'reduce' and os.path.isfile(reduced_path):
        with open(reduced_path, 'rb') as rpo:
            embedding_table = pickle.load(rpo)
        loaded_from_file = True
    elif preprocess == 'whiten' and os.path.isfile(whitened_path):
        with open(whitened_path, 'rb') as wpo:
            embedding_table = pickle.load(wpo)
        loaded_from_file = True

    if loaded_from_file:
        with open(token_path, 'r', encoding='utf8') as tp:
            tokens = json.load(tp)
    else:
        logging.info('Reading-in raw embeddings ... ')
        file_in = io.open(file_path, 'r', encoding='utf8', newline='\n', errors='ignore')
        tokens = list()
        embeddings = list()
        for line_id, line in enumerate(file_in):
            if line_id == 0:
                continue
            splits = line.rstrip().split(' ')
            tokens.append(splits[0])
            embeddings.append(np.array([float(num) for num in splits[1:]], dtype=np.float32))
            if line_id % 100000 == 0:
                logging.info('Added {:d} embeddings'.format(line_id))

        # Preprocess
        embedding_table = np.stack(embeddings, axis=0)
        if preprocess == 'center' or preprocess == 'whitened':
            embedding_table -= np.mean(embedding_table, axis=0)
            pickle_path = centered_path
            logging.info('Centered embeddings!')
        if preprocess == 'reduce':
            cov = np.dot(embedding_table.T, embedding_table) / embedding_table.shape[0]
            u, s, v = np.linalg.svd(cov)
            embedding_table = np.dot(embedding_table, u[:, :10])
            pickle_path = reduced_path
            logging.info('Centered embeddings and reduced their dimensionality!')
        if preprocess == 'whiten':
            cov = np.dot(embedding_table.T, embedding_table) / embedding_table.shape[0]
            u, s, v = np.linalg.svd(cov)
            decor = np.dot(embedding_table, u)
            embedding_table = decor / np.sqrt(s + 1e-5)
            pickle_path = whitened_path
            logging.info('Whitened embeddings!')

        # Dump to file
        with open(pickle_path, 'wb') as pp:
            if pickle_path == whitened_path:
                pickle.dump(embedding_table, pp, protocol=4)
            else:
                pickle.dump(embedding_table, pp)
        with open(token_path, 'w', encoding='utf8') as tp:
            json.dump(tokens, tp, indent=3, sort_keys=True, ensure_ascii=False)

    # Build dict
    logging.info('Building the embedding dictionary ...')
    embeddings = np.vsplit(embedding_table, embedding_table.shape[0])
    for tok_id, tok in enumerate(tokens):
        embedding_dict[tok] = embeddings[tok_id]

    logging.info('Done.')
    logging.info('-' * 20)
    return embedding_dict


def _lookup_embedding(embedding_table, word_token, word_lemma, verbose=False):
    """ Helper function used to lookup word embeddings in the embedding table; incorporates back-off strategies if
    the exact token is not found in embedding table keys. """
    # Map None to None
    if not word_token:
        return None
    # Back-off hierarchy: token -> lower-cased token -> lemma -> None
    if word_token not in embedding_table.keys():
        if word_token.lower() not in embedding_table.keys():
            if word_lemma not in embedding_table.keys():
                if verbose:
                    logging.info('No embedding found for key {:s}'.format(word_token))
                return None
            else:
                return embedding_table[word_lemma]
        else:
            return embedding_table[word_token.lower()]
    else:
        return embedding_table[word_token]


def _get_contextualized_embeddings(sequence, embed_method):
    """ Encodes the provided word string as a sequence of contextualized embeddings. """
    # Return early if using universal sentence encoder
    if embed_method == 'uni_se':
        return model([sequence])
    if embed_method == 'sent_bert':
        return model.encode([sequence])
    # Encode text
    if type(sequence) == list:
        sequence = ' '.join(sequence)
    # Process
    input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True)])
    # Get embeddings
    with torch.no_grad():
        model_outputs = model(input_ids)
    all_hidden_layers = model_outputs[2]
    # NOTE: It doesn't seem to matter whether special tokens are excluded from pooling or not
    if embed_method == 'last_four':
        last_four_layers = [np.squeeze(layer_hs.numpy(), axis=0) for layer_hs in all_hidden_layers[-4:]]
        last_four_time_means = [np.mean(layer_hs, axis=0) for layer_hs in last_four_layers]
        return np.concatenate(last_four_time_means, axis=0)
    else:
        last_layer = np.squeeze(all_hidden_layers[-1].numpy(), axis=0)
        return np.mean(last_layer, axis=0)


def _get_tokens_and_lemmas(line, get_lemmas, nlp):
    """ Helper function for obtaining word tokens and lemmas for the computation of the overlap between two strings. """
    # strip, lowercase, remove punctuation, tokenize, lemmatize
    line_clean = line.strip().strip(punctuation_plus_space).strip()
    # Tokenize etc.
    line_nlp = nlp(line_clean)
    line_tokens = [elem.text for elem in line_nlp]
    line_lemmas = None
    if get_lemmas:
        line_lemmas = list()
        for elem in line_nlp:
            if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
                line_lemmas.append(elem.lower_)
            else:
                line_lemmas.append(elem.lemma_.lower().strip())
    return line_nlp, line_tokens, line_lemmas


def _remove_stopwords(line, nlp):
    """ Helper function for removing stopwords from the given text line. """
    line_nlp = nlp(line)
    line_tokens = [tok.text for tok in line_nlp]
    filtered_line = list()
    # Filter
    for tok in line_tokens:
        lexeme = nlp.vocab[tok]
        if not lexeme.is_stop:
            filtered_line.append(tok)
    return ' '.join(filtered_line)


def _get_polysemous_words(top_n, nouns_only):
    """ Helper function used to collect the top_n polysemous nouns from WordNet,
    ordered according to the number of senses. """
    logging.info('Collecting polysemous terms from WordNet ... ')
    # Collect nouns
    wn_nouns = dict()
    all_synsets = wn.all_synsets(wn.NOUN) if nouns_only else wn.all_synsets()
    for synset in list(all_synsets):
        # Optionally exclude nouns
        if not nouns_only:
            if synset.name().split('.')[1] == 'n':
                continue
        noun = synset.name().split('.')[0]
        # Exclude multi-word nouns
        sub_words = re.sub(r' +', ' ', noun.translate(pct_stripper)).split()
        if len(sub_words) > 1:
            continue
        synset_id = synset.name().split('.')[-1]
        if wn_nouns.get(noun, None) is None:
            wn_nouns[noun] = dict()
        # Exclude synsets likely denoting named entities
        if synset.definition() == synset.definition().lower():
            wn_nouns[noun][synset_id] = synset.definition()
    # Sort nouns by number of synsets and isolate the top-n
    nouns_with_num_synsets = [(noun, len(wn_nouns[noun].keys())) for noun in wn_nouns.keys()]
    sorted_nouns = sorted(nouns_with_num_synsets, reverse=True, key=lambda x: x[1])
    return [tpl[0].strip().lower() for tpl in sorted_nouns][:top_n]


def scrape_senses(src_term_list,
                  src_list_id,
                  src_lang,
                  tgt_lang,
                  output_path):
    """ Scrapes BabelNet sense synsets, de-duplicating them, and filtering out low-quality sense entries. """

    # Instantiate a BabelNet object and sense table
    bn = None  # insert your own key
    # Construct output path
    logging.info('Scraping BabelNet ...')
    logging.info('Currently processing the \'{:s}\' input list ... '.format(src_list_id))
    if os.path.isfile(output_path):
        with open(output_path, 'r', encoding='utf8') as in_fo:
            bn_clusters = json.load(in_fo)
        logging.info('Partial sense-map loaded; continuing collecting sense clusters from last point of interruption.')
    else:
        bn_clusters = dict()
        logging.info('Initializing a new sense map.')

    # Collect senses
    for term_id, term in enumerate(src_term_list):
        logging.info('Looking up senses for the source language term \'{:s}\''.format(term))
        try:
            # Skip previously looked-up terms
            if term in bn_clusters.keys():
                continue
            ids = bn.getSynset_Ids(term, src_lang)
            # Skip empty entries
            if len(ids) == 0:
                logging.info('Non synsets found for \'{:s}\''.format(term))
                continue
            # Extend sense map and avoid clashes
            if not bn_clusters.get(term, None):
                bn_clusters[term] = dict()
            else:
                logging.info('Avoided adding a duplicate entry for \'{:s}\''.format(term))
                continue
            # Iterate over synsets
            sense_clusters = list()
            for id_entry in ids:
                synset_pos = id_entry['pos']
                if src_list_id == 'nouns' and synset_pos != 'NOUN':
                    continue
                synsets = bn.getSynsets(id_entry.id, [src_lang, tgt_lang], change_lang=True)
                # POS, synonyms, senses, src_glosses, tgt_glosses
                curr_sense_cluster = [synset_pos, list(), list(), list(), list()]
                # Scan synset entries
                for synset in synsets:
                    if synset['synsetType'] != 'CONCEPT':
                        continue
                    # Scan retrieved senses
                    for sense in synset['senses']:
                        # Extend current cluster; 'WIKITR' / 'WNTR' translations tend to be source copies;
                        # 'WIKIRED' senses are semantically related but often not direct translations
                        if sense['properties']['source'] not in ['WIKITR', 'WNTR', 'WIKIRED']:
                            if sense['properties']['language'] == src_lang:
                                # Exclude the term itself
                                maybe_syn = sense['properties']['simpleLemma']
                                if maybe_syn.strip().lower() != term.strip().lower():
                                    curr_sense_cluster[1].append(sense['properties']['simpleLemma'])
                            if sense['properties']['language'] == tgt_lang:
                                curr_sense_cluster[2].append(sense['properties']['simpleLemma'])
                    # Retrieve glosses
                    for gloss_obj in synset['glosses']:
                        if gloss_obj['language'] == src_lang:
                            curr_sense_cluster[3].append((gloss_obj['gloss'], gloss_obj['source']))
                        if gloss_obj['language'] == tgt_lang:
                            curr_sense_cluster[4].append((gloss_obj['gloss'], gloss_obj['source']))
                    # De-duplicate glosses
                    curr_sense_cluster[3] = list(set(curr_sense_cluster[3]))
                    curr_sense_cluster[4] = list(set(curr_sense_cluster[4]))
                # Extend cluster list
                if len(curr_sense_cluster[2]) > 0:
                    sense_clusters.append(curr_sense_cluster)
        except Exception as e:
            logging.info('Encountered exception: {:s}'.format(e))
            break

        if len(sense_clusters) > 0:
            # De-duplicate synonyms and senses
            for sc_id, sc in enumerate(sense_clusters):
                unique_synonyms = dict()
                for syn in sc[1]:
                    if not unique_synonyms.get(syn.lower(), None):
                        unique_synonyms[syn.lower()] = re.sub('\\n', '_', syn)
                unique_synonyms = list(unique_synonyms.values())

                unique_senses = dict()
                for sense in sc[2]:
                    if not unique_senses.get(sense.lower(), None):
                        unique_senses[sense.lower()] = re.sub('\\n', '_', sense)
                unique_senses = list(unique_senses.values())
                if len(unique_senses) > 0:
                    bn_clusters[term][sc_id] = [sc[0], unique_synonyms, unique_senses, sc[3], sc[4]]

        # Dump partial table to JSON
        with open(output_path, 'w', encoding='utf8') as out_fo:
            json.dump(bn_clusters, out_fo, indent=3, sort_keys=True, ensure_ascii=False)

        # Report occasionally -- not very efficient
        if term_id > 0 and term_id % 10 == 0:
            total_number_of_clusters = 0
            total_number_of_senses = 0
            for t in bn_clusters.keys():
                total_number_of_clusters += len(bn_clusters[t])
                for s in bn_clusters[t].keys():
                    total_number_of_senses += len(bn_clusters[t][s][2])
            logging.info('Looked-up {:d} polysemous source terms; last one was \'{:s}\''.format(term_id, term))
            logging.info('Average number of sense clusters per term = {:.3f}'
                         .format(total_number_of_clusters / term_id))
            logging.info('Average number of senses per cluster = {:.3f}'
                         .format(total_number_of_senses / total_number_of_clusters))

    # Final report
    logging.info('Done!')
    total_number_of_clusters = 0
    total_number_of_senses = 0
    for t in bn_clusters.keys():
        total_number_of_clusters += len(bn_clusters[t])
        for s in bn_clusters[t].keys():
            total_number_of_senses += len(bn_clusters[t][s][2])
    logging.info('Average number of sense clusters per term = {:.3f}'.
                 format(total_number_of_clusters / len(bn_clusters.keys())))
    logging.info('Average number of senses per cluster = {:.3f}'.
                 format(total_number_of_senses / total_number_of_clusters))


def refine_clusters(bn_senses_path,
                    cluster_similarity_threshold,
                    output_path,
                    use_deep_similarity,
                    use_all_glosses,
                    deep_embed_method,
                    ignore_multiword_senses):
    """ Merges similar BabelNet clusters into super-clusters, either including or ignoring multi-word senses. """

    # Load scraped BabelNet sense clusters
    with open(bn_senses_path, 'r', encoding='utf8') as in_fo:
        no_dup_clusters = json.load(in_fo)

    # Optionally filter out multi-word senses
    if ignore_multiword_senses:
        sense_filtered_clusters = dict()
        logging.info('Refining BabelNet clusters while ignoring multi-word senses ...')
        for term in no_dup_clusters.keys():
            for cs_id in no_dup_clusters[term].keys():
                filtered_senses = [sns for sns in no_dup_clusters[term][cs_id][2] if
                                   '_' not in sns and '-' not in sns]
                if len(filtered_senses) > 0:
                    if not sense_filtered_clusters.get(term, None):
                        sense_filtered_clusters[term] = dict()
                    if not sense_filtered_clusters[term].get(cs_id, None):
                        sense_filtered_clusters[term][cs_id] = no_dup_clusters[term][cs_id][:]
                    sense_filtered_clusters[term][cs_id][2] = filtered_senses
            # Do not include term entries for which no single-word senses could be found
            if not sense_filtered_clusters.get(term, None):
                logging.info('Removed entry for \'{:s}\'; no single-word target senses found.'.format(term))
        no_dup_clusters = sense_filtered_clusters
    else:
        logging.info('Refining BabelNet clusters ...')

    # Remove source / target glosses shorter than three words
    gloss_filtered_clusters = dict()
    for term in no_dup_clusters.keys():
        term_entry = {key: value for (key, value) in no_dup_clusters[term].items()}
        for sense in no_dup_clusters[term].keys():
            filtered_src_glosses = list()
            filtered_tgt_glosses = list()
            for src_gloss_tpl in no_dup_clusters[term][sense][3]:
                src_gloss = re.sub(r' +', ' ', src_gloss_tpl[0].translate(pct_stripper))
                if len(src_gloss.split()) > 3:
                    filtered_src_glosses.append(src_gloss_tpl)
            term_entry[sense][3] = filtered_src_glosses
            for tgt_gloss_tpl in no_dup_clusters[term][sense][4]:
                tgt_gloss = re.sub(r' +', ' ', tgt_gloss_tpl[0].translate(pct_stripper))
                if len(tgt_gloss.split()) > 3:
                    filtered_tgt_glosses.append(tgt_gloss_tpl)
            term_entry[sense][4] = filtered_tgt_glosses
        gloss_filtered_clusters[term] = term_entry
    no_dup_clusters = gloss_filtered_clusters

    # Try to avoid plurals and other inflected forms
    for term in no_dup_clusters.keys():
        for cs_id in no_dup_clusters[term].keys():
            lemma_to_sense = dict()
            for sense in no_dup_clusters[term][cs_id][2]:
                # Get sense lemma
                rep, _, lemmas = _get_tokens_and_lemmas(sense, True, tgt_nlp)
                if len(rep) == 1:
                    sense_lemma = lemmas[0]
                    if lemma_to_sense.get(sense_lemma, None) is None:
                        lemma_to_sense[sense_lemma] = sense
                    else:
                        if len(sense) < len(lemma_to_sense[sense_lemma]):
                            lemma_to_sense[sense_lemma] = sense
            sense_lemmas = [lem for lem in lemma_to_sense.keys()]
            noun_senses = [lemma_to_sense[lem] for lem in sense_lemmas]
            no_dup_clusters[term][cs_id][2] = noun_senses
            no_dup_clusters[term][cs_id].append(sense_lemmas)

    # Remove target sense clusters which are deemed too specific
    info_filtered_clusters = dict()
    clusters_kept = 0
    clusters_dropped = 0
    for term in no_dup_clusters.keys():
        for cs_id in no_dup_clusters[term].keys():
            num_senses = len(no_dup_clusters[term][cs_id][2])
            num_src_glosses = len(no_dup_clusters[term][cs_id][3])
            if num_senses == 0 or num_src_glosses == 0:
                clusters_dropped += 1
                continue
            else:
                if not info_filtered_clusters.get(term, None):
                    info_filtered_clusters[term] = dict()
                info_filtered_clusters[term][cs_id] = no_dup_clusters[term][cs_id]
                clusters_kept += 1

        # Remove term if it only has one sense cluster
        if info_filtered_clusters.get(term, None):
            num_clusters = len(info_filtered_clusters[term])
            if num_clusters < 2:
                clusters_kept -= num_clusters
                clusters_dropped += num_clusters
                info_filtered_clusters.pop(term)

    # Report casualties
    for term in no_dup_clusters.keys():
        if term not in info_filtered_clusters.keys():
            logging.info('Filtered out BabelNet entry for term \'{:s}\''.format(term))
    logging.info('Filtered out {:d} sense clusters in total, kept {:d}'.format(clusters_dropped, clusters_kept))

    no_dup_clusters = info_filtered_clusters

    sense_map = dict()
    for term_id, term in enumerate(no_dup_clusters.keys()):
        logging.info('Refining target sense clusters for source term \'{:s}\''.format(term))
        no_dup_entry = no_dup_clusters[term]
        if not sense_map.get(term, None):
            sense_map[term] = dict()
        # From collected clusters, isolate polysemous target senses (i.e. senses that occur in multiple clusters)
        sense_counts = dict()
        for sense_cluster in no_dup_entry.values():
            for sense in sense_cluster[2]:
                sense_words = sense.split('_')
                for sense_word in sense_words:
                    sns = sense_word.lower()
                    if not sense_counts.get(sns, None):
                        sense_counts[sns] = 1
                    else:
                        sense_counts[sns] += 1

        # Merge overlapping clusters
        logging.info('Looking up sense / gloss information ...')
        # 1. Look up pre-trained word embeddings for each cluster's glosses
        cluster_sense_info = dict()
        for sc_id in no_dup_entry.keys():
            # Construct a lookup table categorizing glosses by their respective DBs
            db_to_gloss = dict()
            for gloss in no_dup_entry[sc_id][3]:
                if not db_to_gloss.get(gloss[1], None):
                    db_to_gloss[gloss[1]] = list()
                db_to_gloss[gloss[1]].append(gloss[0])
            # Initialize cluster info entry ;
            # contains POS, source gloss embeddings, source glosses, target glosses, target senses, source synonyms,
            # target sense lemmas
            cluster_sense_info[sc_id] = [no_dup_entry[sc_id][0], list(), list(),
                                         [tgl[0] for tgl in no_dup_entry[sc_id][4]],
                                         [sns for sns in no_dup_entry[sc_id][2]],
                                         [syn for syn in no_dup_entry[sc_id][1]],
                                         [lem for lem in no_dup_entry[sc_id][5]]]

            # Designate gloss set
            gloss_set = None
            # Optionally prioritize WN / WIKI glosses
            if not use_all_glosses:
                # Gloss prioritization:
                # 1. If set of glosses contains WN glosses, only use those
                # 2. If no WN glosses are found, use WIKI glosses
                # 3. If No WN or WIKI glosses are found, use all glosses
                if 'WN' in db_to_gloss.keys():
                    gloss_set = db_to_gloss['WN']
                elif 'WIKI' in db_to_gloss.keys():
                    gloss_set = db_to_gloss['WIKI']
            # Alternatively, use all available glosses
            if not gloss_set:
                gloss_set = list()
                for db_glosses in db_to_gloss.values():
                    gloss_set += db_glosses

            # Embed glosses
            for gloss in gloss_set:
                if use_deep_similarity:
                    # Use contextualized embeddings
                    gloss_embedding = _get_contextualized_embeddings(gloss, deep_embed_method)
                    cluster_sense_info[sc_id][1].append(gloss_embedding)
                cluster_sense_info[sc_id][2].append(gloss)

        # 2. Iteratively merge clusters
        logging.info('Merging sense clusters ...')
        cluster_merges = dict()
        merged_previous_iter = True
        cluster_ids = list(cluster_sense_info.keys())
        iter_count = 0
        while merged_previous_iter:
            iter_count += 1
            logging.info('Iteration {:d}'.format(iter_count))
            merged_this_round = list()
            merged_previous_iter = False
            # Compute pairwise synset similarity
            pairwise_sims = dict()
            for idx, sc1_id in enumerate(cluster_ids):
                # Continue if cluster as already been merged into another
                if not cluster_sense_info.get(sc1_id, None):
                    continue
                # Continue if cluster has no source glosses or target senses
                if len(cluster_sense_info[sc1_id][2]) == 0 or len(cluster_sense_info[sc1_id][4]) == 0:
                    continue
                for sc2_id in cluster_ids[idx:]:
                    # Continue if cluster as already been merged into another
                    if not cluster_sense_info.get(sc2_id, None):
                        continue
                    # Continue if cluster has no source glosses or target senses
                    if len(cluster_sense_info[sc2_id][2]) == 0 or len(cluster_sense_info[sc2_id][4]) == 0:
                        continue
                    if sc1_id == sc2_id:
                        continue
                    # Only merge clusters with the same POS
                    if cluster_sense_info[sc1_id][0] != cluster_sense_info[sc2_id][0]:
                        continue

                    sim_score = 0.
                    if len(cluster_sense_info[sc1_id][6]) == len(cluster_sense_info[sc2_id][6]) > 2:
                        if sorted(cluster_sense_info[sc1_id][6]) == sorted(cluster_sense_info[sc2_id][6]):
                            sim_score = 1.
                            logging.info('SENSE IDENTITY MERGE!')
                            logging.info('CLUSTER 1: {}'.format(cluster_sense_info[sc1_id][6]))
                            logging.info('CLUSTER 2: {}'.format(cluster_sense_info[sc2_id][6]))

                    if sim_score == 0:
                        # Check if cluster source glosses overlap
                        sc1_src_gloss_strings = [re.sub(r' +', ' ', gl.strip().lower().translate(pct_stripper).strip())
                                                 for gl in cluster_sense_info[sc1_id][2]]
                        sc2_src_gloss_strings = [re.sub(r' +', ' ', gl.strip().lower().translate(pct_stripper).strip())
                                                 for gl in cluster_sense_info[sc2_id][2]]
                        if len(list(set(sc1_src_gloss_strings) & set(sc2_src_gloss_strings))) > 0:
                            sim_score = 1.
                            logging.info('FULL SOURCE GLOSS OVERLAP MERGE!')
                            logging.info('CLUSTER 1: {}'.format(cluster_sense_info[sc1_id][2]))
                            logging.info('CLUSTER 2: {}'.format(cluster_sense_info[sc2_id][2]))

                    if sim_score == 0:
                        # Check if cluster target glosses overlap
                        sc1_tgt_gloss_strings = \
                            [re.sub(r' +', ' ', gl.strip().lower().translate(pct_stripper).strip())
                             for gl in cluster_sense_info[sc1_id][3]]
                        sc2_tgt_gloss_strings = \
                            [re.sub(r' +', ' ', gl.strip().lower().translate(pct_stripper).strip())
                             for gl in cluster_sense_info[sc2_id][3]]
                        if len(list(set(sc1_tgt_gloss_strings) & set(sc2_tgt_gloss_strings))) > 0:
                            sim_score = 1.
                            logging.info('FULL TARGET GLOSS OVERLAP MERGE!')
                            logging.info('CLUSTER 1: {}'.format(cluster_sense_info[sc1_id][3]))
                            logging.info('CLUSTER 2: {}'.format(cluster_sense_info[sc2_id][3]))

                    if sim_score == 0.:
                        sense_overlap_size = \
                            len(list(set(cluster_sense_info[sc1_id][4]) & set(cluster_sense_info[sc2_id][4])))
                        if sense_overlap_size >= 3:
                            sim_score = 1.
                            logging.info('SENSE OVERLAP MERGE!')
                            logging.info('CLUSTER 1: {}'.format(cluster_sense_info[sc1_id][4]))
                            logging.info('CLUSTER 2: {}'.format(cluster_sense_info[sc2_id][4]))

                    if sim_score >= cluster_similarity_threshold:
                        sim_key = tuple(sorted([sc1_id, sc2_id]))
                        if not pairwise_sims.get(sim_key, None):
                            pairwise_sims[sim_key] = sim_score

            # Merge clusters with highest similarity
            pairwise_sims = sorted(list(pairwise_sims.items()), reverse=True, key=lambda x: x[1])
            for ps in pairwise_sims:
                c1_id, c2_id = ps[0]
                # Check if clusters have been merged this round
                if c1_id in merged_this_round or c2_id in merged_this_round:
                    continue

                # Check if either cluster is used as a key in cluster_merges
                cluster_merges_into_c1 = cluster_merges.get(c1_id, None)
                cluster_merges_into_c2 = cluster_merges.get(c2_id, None)
                if not (cluster_merges_into_c1 or cluster_merges_into_c2):
                    cluster_merges[c1_id] = [c2_id]
                    parent_id = c1_id
                    child_id = c2_id
                elif cluster_merges_into_c1:
                    cluster_merges[c1_id].append(c2_id)
                    parent_id = c1_id
                    child_id = c2_id
                else:
                    cluster_merges[c2_id].append(c1_id)
                    parent_id = c2_id
                    child_id = c1_id

                # Merge cluster representations
                # Cache the state of the c1 cluster prior to the merge
                orig_parent_src_glosses = [gl for gl in cluster_sense_info[parent_id][2]]
                orig_parent_tgt_glosses = [gl for gl in cluster_sense_info[parent_id][3]]
                orig_parent_senses = [sns for sns in cluster_sense_info[parent_id][4]]
                # Expand and de-duplicate gloss embeddings
                for emb2 in cluster_sense_info[child_id][1]:
                    is_duplicate = False
                    for emb1 in cluster_sense_info[parent_id][1]:
                        if np.array_equal(emb2, emb1):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        cluster_sense_info[parent_id][1].append(emb2)
                # Expand and de-duplicate source and target glosses
                cluster_sense_info[parent_id][2] += cluster_sense_info[child_id][2]
                cluster_sense_info[parent_id][2] = list(set(cluster_sense_info[parent_id][2]))
                cluster_sense_info[parent_id][3] += cluster_sense_info[child_id][3]
                cluster_sense_info[parent_id][3] = list(set(cluster_sense_info[parent_id][3]))
                # Expand and de-duplicate senses
                cluster_sense_info[parent_id][4] += cluster_sense_info[child_id][4]
                cluster_sense_info[parent_id][4] = list(set(cluster_sense_info[parent_id][4]))
                # Update round-specific merge tracker
                merged_this_round += [parent_id, child_id]
                # Reset flag
                merged_previous_iter = True

                # Report
                logging.info('=' * 20)
                logging.info('Merged clusters {:s} and {:s}'.format(parent_id, child_id))
                logging.info('-' * 10)
                logging.info('PARENT cluster senses: {}'.format(orig_parent_senses))
                logging.info('PARENT cluster source glosses: {}'.format(orig_parent_src_glosses))
                logging.info('PARENT cluster target glosses: {}'.format(orig_parent_tgt_glosses))
                logging.info('-' * 10)
                logging.info('CHILD cluster senses: {}'.format(cluster_sense_info[child_id][4]))
                logging.info('CHILD cluster source glosses: {}'.format(cluster_sense_info[child_id][2]))
                logging.info('CHILD cluster target glosses: {}'.format(cluster_sense_info[child_id][3]))
                logging.info('-' * 10)
                logging.info('PRODUCT cluster senses: {}'.format(cluster_sense_info[parent_id][4]))
                logging.info('PRODUCT cluster source glosses: {}'.format(cluster_sense_info[parent_id][2]))
                logging.info('PRODUCT cluster target glosses: {}'.format(cluster_sense_info[parent_id][3]))
                logging.info('=' * 20)

                # Remove the merged cluster entry
                cluster_sense_info.pop(child_id)

                # Remove parent entry if empty
                if len(cluster_merges[parent_id]) == 0:
                    cluster_merges.pop(parent_id)

        # Add unmerged clusters to the final merged clusters map, for simplicity
        assigned_clusters = list()
        for v in cluster_merges.values():
            assigned_clusters += v
        unmerged_clusters = [sc_id for sc_id in no_dup_entry.keys() if
                             (sc_id not in cluster_merges.keys() and
                              sc_id not in assigned_clusters)]

        # Post process the cluster merge map
        links_found = True
        while links_found:
            linked_merges = dict()
            absorbed = list()
            for pc1 in cluster_merges.keys():
                if pc1 in absorbed:
                    continue
                for pc2 in cluster_merges.keys():
                    if pc2 in absorbed or pc2 == pc1:
                        continue
                    if pc2 in cluster_merges[pc1]:
                        linked_merges[pc1] = list(set(cluster_merges[pc1] + cluster_merges[pc2]))
                        absorbed.append(pc2)
                        # Remove if previously added
                        if linked_merges.get(pc2, None):
                            linked_merges.pop(pc2)
                # Add if left unmodified
                if not linked_merges.get(pc1, None):
                    linked_merges[pc1] = cluster_merges[pc1]
            cluster_merges = linked_merges
            links_found = len(absorbed) > 0

        # Extend sense map
        logging.info('Updating sense map ...')
        for parent_id in cluster_merges.keys():
            sense_map[term][parent_id] = dict()
            sense_map[term][parent_id]['[POS]'] = no_dup_entry[parent_id][0]
            final_cluster_synonyms = no_dup_entry[parent_id][1]
            final_cluster_senses = no_dup_entry[parent_id][2]
            final_cluster_src_glosses = [tuple(gloss) for gloss in no_dup_entry[parent_id][3]]
            final_cluster_tgt_glosses = [tuple(gloss) for gloss in no_dup_entry[parent_id][4]]
            # Accumulate senses and glosses
            for child_id in cluster_merges[parent_id]:
                final_cluster_synonyms += no_dup_entry[child_id][1]
                final_cluster_senses += no_dup_entry[child_id][2]
                final_cluster_src_glosses += [tuple(gloss) for gloss in no_dup_entry[child_id][3]]
                final_cluster_tgt_glosses += [tuple(gloss) for gloss in no_dup_entry[child_id][4]]
            # Assign and deduplicate
            sense_map[term][parent_id]['[SYNONYMS]'] = list(set(final_cluster_synonyms))
            normalized_syns = list()
            for syn in sense_map[term][parent_id]['[SYNONYMS]']:
                if syn[1:] == syn[1:].lower():
                    normalized_syns.append(syn[0].lower() + syn[1:])
            sense_map[term][parent_id]['[SYNONYMS]'] = normalized_syns
            sense_map[term][parent_id]['[SENSES]'] = list(set(final_cluster_senses))
            sense_map[term][parent_id]['[SOURCE GLOSSES]'] = list(set(final_cluster_src_glosses))
            sense_map[term][parent_id]['[TARGET GLOSSES]'] = list(set(final_cluster_tgt_glosses))
        # Add entries for unmodified clusters
        for unmerged_id in unmerged_clusters:
            sense_map[term][unmerged_id] = dict()
            sense_map[term][unmerged_id]['[POS]'] = no_dup_entry[unmerged_id][0]
            sense_map[term][unmerged_id]['[SYNONYMS]'] = list(set(no_dup_entry[unmerged_id][1]))
            normalized_syns = list()
            for syn in sense_map[term][unmerged_id]['[SYNONYMS]']:
                if syn[1:] == syn[1:].lower():
                    normalized_syns.append(syn[0].lower() + syn[1:])
            sense_map[term][unmerged_id]['[SYNONYMS]'] = normalized_syns
            sense_map[term][unmerged_id]['[SENSES]'] = list(set(no_dup_entry[unmerged_id][2]))
            sense_map[term][unmerged_id]['[SOURCE GLOSSES]'] = \
                list(set([tuple(gloss) for gloss in no_dup_entry[unmerged_id][3]]))
            sense_map[term][unmerged_id]['[TARGET GLOSSES]'] = \
                list(set([tuple(gloss) for gloss in no_dup_entry[unmerged_id][4]]))

        # Dump partial table to JSON
        with open(output_path, 'w', encoding='utf8') as out_fo:
            json.dump(sense_map, out_fo, indent=3, sort_keys=True, ensure_ascii=False)

        # Report occasionally -- not very efficient
        if term_id > 0 and term_id % 10 == 0:
            total_number_of_clusters = 0
            total_number_of_senses = 0
            for t in sense_map.keys():
                total_number_of_clusters += len(sense_map[t])
                for s in sense_map[t].keys():
                    total_number_of_senses += len(sense_map[t][s]['[SENSES]'])
            logging.info('Looked-up {:d} polysemous source terms; last one was \'{:s}\''.format(term_id, term))
            logging.info('Average number of sense clusters per term = {:.3f}'
                         .format(total_number_of_clusters / term_id))
            logging.info('Average number of senses per cluster = {:.3f}'
                         .format(total_number_of_senses / total_number_of_clusters))

    logging.info('Done!')
    total_number_of_clusters = 0
    total_number_of_senses = 0
    for t in sense_map.keys():
        total_number_of_clusters += len(sense_map[t])
        for s in sense_map[t].keys():
            total_number_of_senses += len(sense_map[t][s]['[SENSES]'])
    logging.info('Average number of sense clusters per term = {:.3f}'.
                 format(total_number_of_clusters / len(sense_map.keys())))
    logging.info('Average number of senses per cluster = {:.3f}'.
                 format(total_number_of_senses / total_number_of_clusters))

    # Dump to JSON
    with open(output_path, 'w', encoding='utf8') as out_fo:
        json.dump(sense_map, out_fo, indent=3, sort_keys=True, ensure_ascii=False)

    # Final report
    logging.info('Done!')
    total_number_of_clusters = 0
    total_number_of_senses = 0
    for t in sense_map.keys():
        total_number_of_clusters += len(sense_map[t])
        for s in sense_map[t].keys():
            total_number_of_senses += len(sense_map[t][s]['[SENSES]'])
    logging.info('Average number of sense clusters per term = {:.3f}'.
                 format(total_number_of_clusters / len(sense_map.keys())))
    logging.info('Average number of senses per cluster = {:.3f}'.
                 format(total_number_of_senses / total_number_of_clusters))
    logging.info('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='prefix of the path to which the scraped sense table will be saved')
    parser.add_argument('--tgt_embeddings_file_path', type=str, default=None,
                        help='path to file containing the pre-trained target language embeddings')
    parser.add_argument('--src_embeddings_file_path', type=str, default=None,
                        help='path to file containing the pre-trained source language embeddings')
    parser.add_argument('--use_deep_similarity', action='store_true',
                        help='toggles whether to use sense / gloss embeddings')
    parser.add_argument('--use_all_glosses', action='store_true',
                        help='toggles whether all available glosses should be used, or if WN and WIKI glosses should '
                             'be prioritized')
    parser.add_argument('--preprocess_embeddings', type=str, choices=['center', 'reduce', 'whiten'], default=None,
                        help='denotes the pre-processing type to be applied to the pre-trained embeddings')
    parser.add_argument('--cluster_similarity_threshold', type=float, default=0.75,
                        help='threshold value for merging target sense clusters')
    parser.add_argument('--deep_embed_method', type=str, choices=['last_four', 'out_mean', 'uni_se', 'sent_bert'],
                        default='uni_se', help='toggles whether to use the <CLS> representation as sentence embedding')
    parser.add_argument('--action', type=str, required=True, choices=['scrape_nouns', 'scrape_words',
                                                                      'refine_nouns_single', 'refine_words_single',
                                                                      'refine_nouns_multi', 'refine_words_multi'],
                        help='defines the action to be preformed by the script')
    parser.add_argument('--lang_pair', type=str, default=None,
                        help='language pair of the bitext; expected format is src-tgt')
    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.output_prefix.split('/')[:-1])
    file_name = args.output_prefix.split('/')[-1]
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
    src_lang_id, tgt_lang_id = args.lang_pair.strip().split('-')
    spacy_map = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
    try:
        src_nlp = spacy.load(spacy_map[src_lang_id], disable=['parser', 'ner', 'textcat'])
        tgt_nlp = spacy.load(spacy_map[tgt_lang_id], disable=['parser', 'ner', 'textcat'])
    except KeyError:
        logging.info('SpaCy does not support the language {:s} or {:s}. Exiting.'.format(src_lang_id, tgt_lang_id))
        sys.exit(0)
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    punctuation_plus_space = string.punctuation + ' ' + '\t' + '\n'

    # Load pre-trained model / tokenizer
    if args.use_deep_similarity:

        from transformers import *
        import tensorflow_hub as hub
        from sentence_transformers import SentenceTransformer

        if not os.path.exists('./transformers_cache'):
            os.makedirs('./transformers_cache')
        # Declare model type
        model_class = RobertaModel
        tokenizer_class = RobertaTokenizer
        pre_trained_weights = 'roberta-large'
        # Initialize tokenizer / deep model
        if args.deep_embed_method not in ['uni_se', 'sent_bert']:
            tokenizer = tokenizer_class.from_pretrained(pre_trained_weights, cache_dir='./transformers_cache')
            model = model_class.from_pretrained(pre_trained_weights, output_hidden_states=True, output_attentions=True)
        else:
            # TODO: Maybe use LASER?
            if args.deep_embed_method == 'uni_se':
                model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
            else:
                model = SentenceTransformer('bert-large-nli-mean-tokens')
                pass

    # Scrape BabelNet
    if args.action == 'scrape_nouns':
        # POLYSEMOUS_NOUNS = _get_polysemous_words(200, True)
        scrape_path = '{:s}_nouns_scrape.json'.format(args.output_prefix)
        scrape_senses(POLYSEMOUS_NOUNS, 'nouns', src_lang_id.upper(), tgt_lang_id.upper(), scrape_path)

    elif args.action == 'scrape_words':
        POLYSEMOUS_WORDS = _get_polysemous_words(200, False)
        scrape_path = '{:s}_words_scrape.json'.format(args.output_prefix)
        scrape_senses(POLYSEMOUS_WORDS, 'words', src_lang_id.upper(), tgt_lang_id.upper(), scrape_path)

    elif args.action == 'refine_nouns_single':
        scrape_path = '{:s}_nouns_scrape.json'.format(args.output_prefix)
        single_clusters_path = '{:s}_nouns_single_senses.json'.format(args.output_prefix)
        refine_clusters(scrape_path,
                        args.cluster_similarity_threshold,
                        single_clusters_path,
                        args.use_deep_similarity,
                        args.use_all_glosses,
                        args.deep_embed_method,
                        ignore_multiword_senses=True)

    elif args.action == 'refine_words_single':
        scrape_path = '{:s}_words_scrape.json'.format(args.output_prefix)
        single_clusters_path = '{:s}_words_single_senses.json'.format(args.output_prefix)
        refine_clusters(scrape_path,
                        args.cluster_similarity_threshold,
                        single_clusters_path,
                        args.use_deep_similarity,
                        args.use_all_glosses,
                        args.deep_embed_method,
                        ignore_multiword_senses=True)

    elif args.action == 'refine_nouns_multi':
        scrape_path = '{:s}_nouns_scrape.json'.format(args.output_prefix)
        single_clusters_path = '{:s}_nouns_multi_senses.json'.format(args.output_prefix)
        refine_clusters(scrape_path,
                        args.cluster_similarity_threshold,
                        single_clusters_path,
                        args.use_deep_similarity,
                        args.use_all_glosses,
                        args.deep_embed_method,
                        ignore_multiword_senses=False)

    else:
        scrape_path = '{:s}_words_scrape.json'.format(args.output_prefix)
        single_clusters_path = '{:s}_words_multi_senses.json'.format(args.output_prefix)
        refine_clusters(scrape_path,
                        args.cluster_similarity_threshold,
                        single_clusters_path,
                        args.use_deep_similarity,
                        args.use_all_glosses,
                        args.deep_embed_method,
                        ignore_multiword_senses=False)

