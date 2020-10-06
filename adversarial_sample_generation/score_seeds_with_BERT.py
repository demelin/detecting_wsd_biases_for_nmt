import os
import json
import torch
import random
import logging
import argparse

import numpy as np
from transformers import *


def evaluate_seeds(seed_sentences_path, sense_table_path, device, num_samples, verbose, incremental_masking=False):
    """ Checks BERT's predictions of the homograph's translation. """

    def _get_bert_score(sample_tokens, _mask_indices, _sense_tokens):
        """ Helper function used to obtain the BERT probability for all masked subword tokens """
        # Encode sample
        sample_ids = tokenizer.convert_tokens_to_ids(sample_tokens)
        sense_token_ids = tokenizer.convert_tokens_to_ids(_sense_tokens)
        sample_ids = tokenizer.encode(sample_ids, add_special_tokens=True)
        segments_ids = [0] * len(sample_ids)
        tokens_tensor = torch.unsqueeze(torch.tensor(sample_ids), 0)
        segments_tensor = torch.unsqueeze(torch.tensor(segments_ids), 0)
        # Move to specified device
        model.eval()
        model.to(device)
        tokens_tensor = tokens_tensor.to(device)
        segments_tensor = segments_tensor.to(device)
        # Do a forward pass
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensor)
        # Softmax logits for better comparability
        predictions = torch.softmax(outputs[0], axis=-1).cpu().numpy()
        # Look up prediction scores for sequence tokens at each time-step
        predictions = np.squeeze(predictions, axis=0)
        # Remove special tokens
        predictions_per_step = np.split(predictions, predictions.shape[0])[1: -1]
        predictions_per_step = [np.squeeze(step, axis=0) for step in predictions_per_step]
        masked_steps = [predictions_per_step[loc] for loc in _mask_indices]
        return [step[sense_token_ids[step_id]] for step_id, step in enumerate(masked_steps)]

    def _get_incremental_bert_score(sample_tokens, mask_location, sense_token):
        """ Helper function used to obtain the BERT probability for a single masked subword token """
        # Encode sample
        sample_ids = tokenizer.convert_tokens_to_ids(sample_tokens)
        sense_token_id = tokenizer.convert_tokens_to_ids(sense_token)
        sample_ids = tokenizer.encode(sample_ids, add_special_tokens=True)
        segments_ids = [0] * len(sample_ids)
        tokens_tensor = torch.unsqueeze(torch.tensor(sample_ids), 0)
        segments_tensor = torch.unsqueeze(torch.tensor(segments_ids), 0)
        # Move to specified device
        model.eval()
        model.to(device)
        tokens_tensor = tokens_tensor.to(device)
        segments_tensor = segments_tensor.to(device)
        # Do a forward pass
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensor)
        # Softmax logits for better comparability
        predictions = torch.softmax(outputs[0], axis=-1).cpu().numpy()
        # Look up prediction scores for sequence tokens at each time-step
        predictions = np.squeeze(predictions, axis=0)
        # Remove special tokens
        predictions_per_step = np.split(predictions, predictions.shape[0])[1: -1]
        predictions_per_step = [np.squeeze(step, axis=0) for step in predictions_per_step]
        masked_step = predictions_per_step[mask_location]
        return masked_step[sense_token_id]

    def _keep_seed(_seed_tpl, _seen_term_seeds):
        """ Helper function used to determine whether to retain or discard the seed tuple """
        if tgt_loc_list is None:
            src_sent, tgt_sent, _, _, _, _, _, _, _, _, ws_tgt_sns_loc = _seed_tpl
        else:
            src_sent, tgt_sent, spacy_src_pos, spacy_tgt_pos, align_line, src_homograph_token, \
                tgt_sns_token = _seed_tpl
            spacy_tgt_sns_loc, ws_tgt_sns_loc = tgt_loc_list[tpl_id]
            spacy_src_loc, ws_src_loc = src_loc_list[tpl_id]
            # Re-build seed tuple for subsequent modifier extraction
            _seed_tpl = (src_sent, tgt_sent, spacy_src_pos, spacy_tgt_pos, align_line, src_homograph_token,
                         tgt_sns_token, spacy_src_loc, ws_src_loc, spacy_tgt_sns_loc, ws_tgt_sns_loc)
        # Report progress in more detail
        if _seen_term_seeds > 0 and _seen_term_seeds % 100 == 0:
            logging.info('Seen {:d} seeds for the current term'.format(_seen_term_seeds))
        _seen_term_seeds += 1

        # Isolate the correct translation
        true_sense = tgt_sent.split()[ws_tgt_sns_loc[0]]
        true_samples = [(tgt_sent, true_sense, True)]
        # Sample additional correct target senses of the source homograph
        all_adv_samples = list()
        true_senses = sense_clusters[term][cluster]['[SENSES]']
        if len(true_senses) > num_samples + 1:
            true_senses = random.sample(true_senses, num_samples + 1)
        if true_sense in true_senses:
            true_senses.pop(true_senses.index(true_sense))
        else:
            if len(true_senses) == num_samples:
                true_senses.pop(0)
        # Sample incorrect target senses of the source homograph (for each sense cluster)
        all_adv_senses = list()
        for adv_cluster in sense_clusters[term].keys():
            if adv_cluster == cluster:
                continue
            adv_senses = sense_clusters[term][adv_cluster]['[SENSES]']
            if len(adv_senses) > num_samples:
                adv_senses = random.sample(adv_senses, num_samples)
            all_adv_senses.append(adv_senses)
        # Generate samples for evaluation
        tgt_ws_tokens = tgt_sent.strip().split()
        tgt_prefix = tgt_ws_tokens[:ws_tgt_sns_loc[0]]
        tgt_suffix = tgt_ws_tokens[ws_tgt_sns_loc[0] + 1:]
        for trs in true_senses:
            true_samples.append((' '.join(tgt_prefix + [trs] + tgt_suffix), trs, True))
        for adv_senses in all_adv_senses:
            adv_samples = list()
            for ads in adv_senses:
                adv_samples.append((' '.join(tgt_prefix + [ads] + tgt_suffix), ads, False))
            if len(adv_samples) > 0:
                all_adv_samples.append(adv_samples)

        # Ignore samples for which no adversarial sets can be constructed
        if len(true_samples) == 0 or max([len(sublist) for sublist in all_adv_samples]) == 0:
            return False, _seed_tpl, _seen_term_seeds
        # Combine sample lists for convenience
        all_samples = list()
        for sample in true_samples:
            all_samples.append(sample)
        for sublist in all_adv_samples:
            for sample in sublist:
                all_samples.append(sample)
        # Track scores
        true_scores = list()
        true_scores_per_token = list()
        adv_scores = list()
        adv_scores_per_token = list()
        tokenized_senses = list()

        for tgt_sample, tgt_sense, tag in all_samples:
            # Tokenize
            tgt_tokens = tokenizer.tokenize(tgt_sample)
            sense_tokens = tokenizer.tokenize(tgt_sense)
            tokenized_senses.append(sense_tokens)
            # Ignore seed translations of excessive length
            if len(tgt_tokens) > 510:
                continue
            # Identify mask-able locations
            locations_to_mask = [[j for j in range(i, i + len(sense_tokens))]
                                 for i in range(len(tgt_tokens) - len(sense_tokens) + 1)
                                 if tgt_tokens[i:i + len(sense_tokens)] == sense_tokens]
            if len(locations_to_mask) > 1:
                all_sense_loc = \
                    [loc for loc, tok in enumerate(tgt_sample.strip().split()) if tok == tgt_sense]
                relevant_sense_instance = all_sense_loc.index(ws_tgt_sns_loc[0])
                mask_indices = locations_to_mask[relevant_sense_instance]
            else:
                mask_indices = locations_to_mask[0]

            # Mask each subword of the target sense separately
            if incremental_masking:
                subword_scores = list()
                for subword_id, sense_subword in enumerate(sense_tokens):
                    masked_tgt_tokens = [tok if tok_loc != mask_indices[subword_id] else '[MASK]' for
                                         tok_loc, tok in enumerate(tgt_tokens)]
                    subword_scores.append(_get_incremental_bert_score(
                        masked_tgt_tokens, mask_indices[subword_id], sense_subword))
            else:
                masked_tgt_tokens = [tok if tok_loc not in mask_indices else '[MASK]' for
                                     tok_loc, tok in enumerate(tgt_tokens)]
                subword_scores = _get_bert_score(masked_tgt_tokens, mask_indices, sense_tokens)

            if len(subword_scores) == 1:
                sense_score = subword_scores[0]
            else:
                sense_score = np.prod(subword_scores) if incremental_masking else subword_scores[0]

            if tag is True:
                true_scores.append(sense_score)
                true_scores_per_token.append(subword_scores)
            else:
                adv_scores.append(sense_score)
                adv_scores_per_token.append(subword_scores)

        # Surface some details
        if verbose:
            logging.info('-' * 20)
            logging.info('DISCARDED')
            logging.info(src_sent.strip())
            logging.info(tgt_sent.strip())
            logging.info('\n')
            logging.info(true_samples)
            logging.info(all_adv_samples)
            logging.info('\n')
            logging.info(tokenized_senses)
            logging.info('\n')
            logging.info(true_scores)
            logging.info(true_scores_per_token)
            logging.info(adv_scores)
            logging.info(adv_scores_per_token)

        # Keep or discard seed tuple
        if len(true_scores) == 0 or len(adv_scores) == 0:
            return False, _seed_tpl, _seen_term_seeds
        return np.max(true_scores) > np.max(adv_scores), _seed_tpl, _seen_term_seeds

    # Read-in seed sentence pairs
    if seed_sentences_path is not None:
        logging.info('Reading-in collected seed sentence pairs ...')
        with open(seed_sentences_path, 'r', encoding='utf8') as chp:
            seed_sentences = json.load(chp)

    # Read-in sense table
    logging.info('Reading-in sense table ...')
    with open(sense_table_path, 'r', encoding='utf8') as scp:
        sense_clusters = json.load(scp)

    # Check file type
    test_key1 = list(seed_sentences.keys())[0]
    test_key2 = list(seed_sentences[test_key1].keys())[0]
    has_forms = type(seed_sentences[test_key1][test_key2]) == dict

    # Track predictions
    kept_seed_sentences = dict()
    dropped_seed_sentences = dict()
    seeds_kept = 0
    seeds_dropped = 0

    # Iterate
    for term_id, term in enumerate(seed_sentences.keys()):
        seen_term_seeds = 0
        # Compute modifier cluster size cut-off to encourage fair comparison
        logging.info('Looking-up the term \'{:s}\''.format(term))
        if has_forms:
            for form in seed_sentences[term].keys():
                for cluster in seed_sentences[term][form].keys():
                    if type(seed_sentences[term][form][cluster]) == list:
                        seed_sentence_list = seed_sentences[term][form][cluster]
                        tgt_loc_list = None
                        src_loc_list = None
                    else:
                        # NOTE: This allows the scoring of sentences used for attractor / modifier extraction
                        seed_sentence_list = seed_sentences[term][form][cluster]['[SENTENCE PAIRS]']
                        tgt_loc_list = seed_sentences[term][form][cluster]['[TARGET TERM LOCATIONS]']
                        src_loc_list = seed_sentences[term][form][cluster]['[SOURCE TERM LOCATIONS]']

                    for tpl_id, seed_tpl in enumerate(seed_sentence_list):
                        keep_seed, seed_tpl, seen_term_seeds = _keep_seed(seed_tpl, seen_term_seeds)

                        # Add seed to the filtered set, if conditions are satisfied
                        if keep_seed:
                            if kept_seed_sentences.get(term, None) is None:
                                kept_seed_sentences[term] = dict()
                            if kept_seed_sentences[term].get(form, None) is None:
                                kept_seed_sentences[term][form] = dict()
                            if kept_seed_sentences[term][form].get(cluster, None) is None:
                                kept_seed_sentences[term][form][cluster] = [seed_tpl]
                            else:
                                kept_seed_sentences[term][form][cluster].append(seed_tpl)
                            if verbose:
                                logging.info('KEPT')
                            seeds_kept += 1
                        # Discard otherwise
                        else:
                            if dropped_seed_sentences.get(term, None) is None:
                                dropped_seed_sentences[term] = dict()
                            if dropped_seed_sentences[term].get(form, None) is None:
                                dropped_seed_sentences[term][form] = dict()
                            if dropped_seed_sentences[term][form].get(cluster, None) is None:
                                dropped_seed_sentences[term][form][cluster] = [seed_tpl]
                            else:
                                dropped_seed_sentences[term][form][cluster].append(seed_tpl)
                            if verbose:
                                logging.info('DISCARD')
                            seeds_dropped += 1

        else:
            for cluster in seed_sentences[term].keys():
                if type(seed_sentences[term][cluster]) == list:
                    seed_sentence_list = seed_sentences[term][cluster]
                    tgt_loc_list = None
                    src_loc_list = None
                else:
                    # NOTE: This allows the scoring of sentences used for attractor / modifier extraction
                    seed_sentence_list = seed_sentences[term][cluster]['[SENTENCE PAIRS]']
                    tgt_loc_list = seed_sentences[term][cluster]['[TARGET TERM LOCATIONS]']
                    src_loc_list = seed_sentences[term][cluster]['[SOURCE TERM LOCATIONS]']

                for tpl_id, seed_tpl in enumerate(seed_sentence_list):
                    keep_seed, seed_tpl, seen_term_seeds = _keep_seed(seed_tpl, seen_term_seeds)

                    # Add seed to the filtered set, if conditions are satisfied
                    if keep_seed:
                        if kept_seed_sentences.get(term, None) is None:
                            kept_seed_sentences[term] = dict()
                        if kept_seed_sentences[term].get(cluster, None) is None:
                            kept_seed_sentences[term][cluster] = [seed_tpl]
                        else:
                            kept_seed_sentences[term][cluster].append(seed_tpl)
                        if verbose:
                            logging.info('KEPT')
                        seeds_kept += 1
                    # Discard otherwise
                    else:
                        if dropped_seed_sentences.get(term, None) is None:
                            dropped_seed_sentences[term] = dict()
                        if dropped_seed_sentences[term].get(cluster, None) is None:
                            dropped_seed_sentences[term][cluster] = [seed_tpl]
                        else:
                            dropped_seed_sentences[term][cluster].append(seed_tpl)
                        if verbose:
                            logging.info('DISCARD')
                        seeds_dropped += 1

        # Incremental reporting
        logging.info('SEEDS KEPT: {:d} | SEEDS DROPPED: {:d}'.format(seeds_kept, seeds_dropped))

    # Save
    out_path = seed_sentences_path
    if out_path.endswith('.json'):
        out_path = out_path[:-5]
    kept_out_path = '{:s}_high_true_bias.json'.format(out_path)
    dropped_out_path = '{:s}_low_true_bias.json'.format(out_path)

    with open(kept_out_path, 'w', encoding='utf8') as kop:
        json.dump(kept_seed_sentences, kop, indent=3, sort_keys=True, ensure_ascii=False)

    with open(dropped_out_path, 'w', encoding='utf8') as dop:
        json.dump(dropped_seed_sentences, dop, indent=3, sort_keys=True, ensure_ascii=False)

    logging.info('Done!')
    logging.info('Seed sentences kept : {:d}'.format(seeds_kept))
    logging.info('Seed sentences dropped : {:d}'.format(seeds_dropped))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_sentences_path', type=str, required=True,
                        help='path to the JSON file containing the extracted seed sentence pairs')
    parser.add_argument('--sense_clusters_path', type=str, required=True,
                        help='path to the JSON file containing scraped BabelNet sense clusters')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                        help='denotes the device used to run the pre-trained language model')
    parser.add_argument('--num_samples', type=int, default=3, help='denotes the number of positive and negative '
                                                                   'samples to be generated for each seed sentence')
    parser.add_argument('--verbose', action='store_true', help='enables additional logging used for debugging')

    args = parser.parse_args()

    # Logging to file
    base_dir = '/'.join(args.seed_sentences_path.split('/')[:-1])
    file_name = args.seed_sentences_path.split('/')[-1]
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

    # Initialize pre-trained LM
    model_class = BertForMaskedLM
    tokenizer_class = BertTokenizer
    pre_trained_weights = 'bert-base-german-dbmdz-cased'
    # Load pre-trained model/tokenizer
    if not os.path.exists('./transformers_cache'):
        os.makedirs('./transformers_cache')
    tokenizer = tokenizer_class.from_pretrained(pre_trained_weights, cache_dir='./transformers_cache')
    # Initialize deep model
    model = model_class.from_pretrained(pre_trained_weights, num_labels=2, output_hidden_states=True,
                                        output_attentions=True, cache_dir='./transformers_cache')

    evaluate_seeds(args.seed_sentences_path,
                   args.sense_clusters_path,
                   args.device,
                   args.num_samples,
                   args.verbose)


