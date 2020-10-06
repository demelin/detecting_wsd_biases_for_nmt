import sys
import json
import torch
import argparse

import numpy as np
import torch.nn.functional as F

from transformers import *


def _get_ppl(sentence):
    """ Produces a sentence-level LM score using a pre-trained LM """
    input_tokens = tokenizer.tokenize(sentence, add_prefix_space=True)
    # input_ids = torch.tensor([tokenizer.encode(input_tokens, add_special_tokens=False)])
    input_ids = torch.unsqueeze(torch.tensor([tokenizer.bos_token_id] +
                                             tokenizer.convert_tokens_to_ids(input_tokens) +
                                             [tokenizer.eos_token_id]), 0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        lm_loss, logits = model(input_ids, labels=input_ids)[:2]
        log_probs = F.log_softmax(logits, dim=-1)
        shifted_input_ids = input_ids[:, 1:]
        nlls = (-log_probs[0, :-1, :].gather(1, shifted_input_ids.view(-1, 1))).view(-1).cpu().numpy().tolist()
        ppls = list(map(lambda nll: np.exp(nll), nlls))
    return input_tokens, ppls, np.mean(ppls), np.exp(lm_loss.cpu().item()), \
        np.exp(lm_loss.cpu().item()) / len(input_tokens)


def score_samples(sample_path, max_ppl_increase=0.2):
    """ Uses a pre-trained language model to assign a fluency score to the generated adversarial samples. """

    # Read-in samples
    logging.info('Reading-in samples to be scored ...')
    with open(sample_path, 'r', encoding='utf8') as sp:
        samples_table = json.load(sp)

    has_forms = None
    # Check data type
    while has_forms is None:
        try:
            test_key1 = list(samples_table.keys())[0]
            test_key2 = list(samples_table[test_key1].keys())[0]
            test_key3 = list(samples_table[test_key1][test_key2].keys())[0]
            has_forms = type(samples_table[test_key1][test_key2][test_key3]) == dict
        except IndexError:
            continue

    # Initialize filtered table
    kept_samples_table = dict()
    dropped_samples_table = dict()

    # Cache seen seed sentence scores
    seed_scores = dict()
    scored_sample_entries = list()

    ordered_terms = sorted(samples_table.keys())
    for term_id, term in enumerate(ordered_terms):
        print('Looking-up the term \'{:s}\''.format(term))
        if has_forms:
            for form in samples_table[term].keys():
                for seed_cluster in samples_table[term][form].keys():
                    for adv_cluster in samples_table[term][form][seed_cluster].keys():
                        scored_samples = 0
                        for sample_tuple in samples_table[term][form][seed_cluster][adv_cluster]:
                            # Isolate sequences
                            adv_source = sample_tuple[0].strip()
                            seed_source = sample_tuple[1].strip()
                            # Score sequences
                            _, _, _, adv_ppl, norm_adv_ppl = _get_ppl(adv_source)
                            if seed_scores.get(seed_source, None) is None:
                                _, _, _, seed_ppl, norm_seed_ppl = _get_ppl(seed_source)
                                seed_scores[seed_source] = norm_seed_ppl
                            norm_seed_ppl = seed_scores[seed_source]
                            ppl_ratio = norm_adv_ppl / norm_seed_ppl
                            sample_tuple = tuple(list(sample_tuple) + [ppl_ratio])
                            scored_sample_entries.append((sample_tuple, term, form, seed_cluster, adv_cluster))
                            scored_samples += 1
                            if scored_samples % 500 == 0:
                                print('Scored {:d} / {:d} samples'
                                      .format(scored_samples,
                                              len(samples_table[term][form][seed_cluster][adv_cluster])))

        else:
            for seed_cluster in samples_table[term].keys():
                for adv_cluster in samples_table[term][seed_cluster].keys():
                    scored_samples = 0
                    for sample_tuple in samples_table[term][seed_cluster][adv_cluster]:
                        # Isolate sequences
                        adv_source = sample_tuple[0].strip()
                        seed_source = sample_tuple[1].strip()
                        # Score sequences
                        _, _, _, adv_ppl, norm_adv_ppl = _get_ppl(adv_source)
                        if seed_scores.get(seed_source, None) is None:
                            _, _, _, seed_ppl, norm_seed_ppl = _get_ppl(seed_source)
                            seed_scores[seed_source] = norm_seed_ppl
                        norm_seed_ppl = seed_scores[seed_source]
                        ppl_ratio = norm_adv_ppl / norm_seed_ppl
                        sample_tuple = tuple(list(sample_tuple) + [ppl_ratio])
                        scored_sample_entries.append((sample_tuple, term, None, seed_cluster, adv_cluster))
                        scored_samples += 1
                        if scored_samples % 500 == 0:
                            print('Scored {:d} / {:d} samples'
                                  .format(scored_samples, len(samples_table[term][seed_cluster][adv_cluster])))


    # Filter out disfluent samples
    kept_samples = list()
    dropped_samples = list()
    for tpl in scored_sample_entries:
        if tpl[0][-1] <= 1.:
            kept_samples.append(tpl)
        else:
            if tpl[0][-1] <= 1. + max_ppl_increase:
                kept_samples.append(tpl)
            else:
                dropped_samples.append(tpl)

    # Integrate
    if len(kept_samples) > 0:
        for tpl in kept_samples:
            sample_tpl, term, form, seed_cluster, adv_cluster = tpl
            if kept_samples_table.get(term, None) is None:
                kept_samples_table[term] = dict()
            if has_forms:
                if kept_samples_table[term].get(form, None) is None:
                    kept_samples_table[term][form] = dict()
                expanded_table = kept_samples_table[term][form]
            else:
                expanded_table = kept_samples_table[term]
            if expanded_table.get(seed_cluster, None) is None:
                expanded_table[seed_cluster] = dict()
            if expanded_table[seed_cluster].get(adv_cluster, None) is None:
                expanded_table[seed_cluster][adv_cluster] = [sample_tpl]
            else:
                expanded_table[seed_cluster][adv_cluster].append(sample_tpl)

    if len(dropped_samples) > 0:
        for tpl in dropped_samples:
            sample_tpl, term, form, seed_cluster, adv_cluster = tpl
            if dropped_samples_table.get(term, None) is None:
                dropped_samples_table[term] = dict()
            if has_forms:
                if dropped_samples_table[term].get(form, None) is None:
                    dropped_samples_table[term][form] = dict()
                expanded_table = dropped_samples_table[term][form]
            else:
                expanded_table = dropped_samples_table[term]
            if expanded_table.get(seed_cluster, None) is None:
                expanded_table[seed_cluster] = dict()
            if expanded_table[seed_cluster].get(adv_cluster, None) is None:
                expanded_table[seed_cluster][adv_cluster] = [sample_tpl]
            else:
                expanded_table[seed_cluster][adv_cluster].append(sample_tpl)

    # Report
    print('LM-filtering completed')
    print('Kept {:d} samples'.format(len(kept_samples)))
    print('Dropped {:d} samples'.format(len(dropped_samples)))

    # Save
    out_path_keep = sample_path[:-5] + '_lm_keep.json'
    out_path_drop = sample_path[:-5] + '_lm_drop.json'
    print('Saving kept adversarial samples to {:s}'.format(out_path_keep))
    with open(out_path_keep, 'w', encoding='utf8') as opk:
        json.dump(kept_samples_table, opk, indent=3, sort_keys=True, ensure_ascii=False)
    print('Saving dropped adversarial samples to {:s}'.format(out_path_drop))
    with open(out_path_drop, 'w', encoding='utf8') as opd:
        json.dump(dropped_samples_table, opd, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, help='path to the file containing the generated adversarial samples',
    args = parser.parse_args()

    # 'openai-gpt' for GPT
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    score_samples(args.sample_path)
