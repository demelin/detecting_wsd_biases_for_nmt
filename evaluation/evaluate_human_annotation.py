import sys
import json
import math
import argparse
import numpy as np

from pingouin import corr
from sklearn.metrics import cohen_kappa_score


def read_tables(form_path_1, key_path_1, form_path_2, key_path_2):
    """ Reads-in human judgements and reports results. """

    # Read-in forms
    form_1 = open(form_path_1, 'r', encoding='utf8')
    form_2 = open(form_path_2, 'r', encoding='utf8')

    # Read in keys
    with open(key_path_1, 'r', encoding='utf8') as kp1:
        keys_1 = json.load(kp1)
    with open(key_path_2, 'r', encoding='utf8') as kp2:
        keys_2 = json.load(kp2)

    # Trackers
    correct_sense_pick = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}
    is_ambiguous = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}
    is_natural = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}

    shared_correct_sense_pick_1 = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}
    shared_is_ambiguous_1 = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}
    shared_is_natural_1 = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}

    shared_correct_sense_pick_2 = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}
    shared_is_ambiguous_2 = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}
    shared_is_natural_2 = {'wmt': {'natural': [], 'synthetic': []}, 'os': {'natural': [], 'synthetic': []}}

    # Go through annotations line by line
    for form, keys, shared_correct_sense_pick, shared_is_ambiguous, shared_is_natural in \
            [(form_1, keys_1, shared_correct_sense_pick_1, shared_is_ambiguous_1, shared_is_natural_1),
             (form_2, keys_2, shared_correct_sense_pick_2, shared_is_ambiguous_2, shared_is_natural_2)]:
        for line_id, line in enumerate(form):
            if line_id < 2:
                continue
            key = keys[str(line_id - 1)]
            domain, prv, sns_1, sns_2 = key
            sns_tpl = (sns_1, sns_2)
            sns_pick, amb_pick, nat_pick = line.split('\t')[-3:]

            # Assign to trackers
            if line_id < 1002:
                if sns_pick in ['BOTH', 'NONE']:
                    pass
                else:
                    correct_sense_pick[domain][prv].append(int(sns_tpl[int(sns_pick) - 1]))
                if amb_pick == 'UNSURE':
                    pass
                else:
                    is_ambiguous[domain][prv].append(int(amb_pick == 'NO'))
                    is_natural[domain][prv].append(int(nat_pick))

            else:
                # Assign to trackers
                shared_correct_sense_pick[domain][prv].append(sns_pick)
                shared_is_ambiguous[domain][prv].append(amb_pick)
                shared_is_natural[domain][prv].append(int(nat_pick))

    # Report summary
    print('Correct sense picked:')
    all_natural = list()
    all_synthetic = list()
    for domain in correct_sense_pick.keys():
        for prv in correct_sense_pick[domain]:
            if prv == 'natural':
                all_natural += correct_sense_pick[domain][prv]
            else:
                all_synthetic += correct_sense_pick[domain][prv]
            total = len(correct_sense_pick[domain][prv])
            pos = sum(correct_sense_pick[domain][prv])
            neg = total - pos
            print('{:s} | {:s} : Yes {:d} ({:.3f}%) | No {:d} ({:.3f}%)'
                  .format(domain, prv, pos, (pos / total) * 100, neg, (neg / total) * 100))
    for tag, scores in [('all natural', all_natural), ('all synthetic', all_synthetic)]:
        total = len(scores)
        pos = sum(scores)
        neg = total - pos
        print('{:s} : Yes {:d} ({:.3f}%) | No {:d} ({:.3f}%)'
              .format(tag, pos, (pos / total) * 100, neg, (neg / total) * 100))
    print('=' * 20)

    print('Homograph is NOT ambiguous:')
    all_natural = list()
    all_synthetic = list()
    for domain in is_ambiguous.keys():
        for prv in is_ambiguous[domain]:
            if prv == 'natural':
                all_natural += is_ambiguous[domain][prv]
            else:
                all_synthetic += is_ambiguous[domain][prv]
            total = len(is_ambiguous[domain][prv])
            pos = sum(is_ambiguous[domain][prv])
            neg = total - pos
            print('{:s} | {:s} : Yes {:d} ({:.3f}%) | No {:d} ({:.3f}%)'
                  .format(domain, prv, pos, (pos / total) * 100, neg, (neg / total) * 100))
    for tag, scores in [('all natural', all_natural), ('all synthetic', all_synthetic)]:
        total = len(scores)
        pos = sum(scores)
        neg = total - pos
        print('{:s} : Yes {:d} ({:.3f}%) | No {:d} ({:.3f}%)'
              .format(tag, pos, (pos / total) * 100, neg, (neg / total) * 100))
    print('=' * 20)

    print('Naturalness scores:')
    all_natural = list()
    all_synthetic = list()
    for domain in is_natural.keys():
        for prv in is_natural[domain]:
            if prv == 'natural':
                all_natural += is_natural[domain][prv]
            else:
                all_synthetic += is_natural[domain][prv]
            print('{:s} | {:s} : {:.3f}'.format(domain, prv, np.mean(is_natural[domain][prv])))
    for tag, scores in [('all natural', all_natural), ('all synthetic', all_synthetic)]:
        print('{:s} : {:.3f}'.format(tag, np.mean(scores)))
    print('=' * 20)

    print('Rater agreement - Cohen\'s (weighted) kappa:')
    all_1 = list()
    all_2 = list()
    print('Correct sense picked:')
    for domain in shared_correct_sense_pick_1.keys():
        for prv in shared_correct_sense_pick_1[domain]:
            all_1 += shared_correct_sense_pick_1[domain][prv]
            all_2 += shared_correct_sense_pick_2[domain][prv]
    ck_sns = cohen_kappa_score(all_1, all_2, labels=['1', '2', 'NONE', 'BOTH'])
    ck_sns = 1. if math.isnan(ck_sns) else ck_sns
    print(ck_sns)

    print('Homograph is NOT ambiguous:')
    all_1 = list()
    all_2 = list()
    for domain in shared_is_ambiguous_1.keys():
        for prv in shared_is_ambiguous_1[domain]:
            all_1 += shared_is_ambiguous_1[domain][prv]
            all_2 += shared_is_ambiguous_2[domain][prv]
    ck_amb = cohen_kappa_score(all_1, all_2, labels=['YES', 'NO', 'UNSURE'])
    ck_amb = 1. if math.isnan(ck_amb) else ck_amb
    print(ck_amb)

    print('Naturalness scores:')
    all_1 = list()
    all_2 = list()
    for domain in shared_is_natural_1.keys():
        for prv in shared_is_natural_1[domain]:
            all_1 += shared_is_natural_1[domain][prv]
            all_2 += shared_is_natural_2[domain][prv]
    ck_nat = cohen_kappa_score(all_1, all_2, labels=[1, 2, 3, 4, 5], weights='linear')
    ck_nat = 1. if math.isnan(ck_nat) else ck_nat
    print(ck_nat)
    print(corr(all_1, all_2, method='pearson').round(3))

    print('Mean agreement: {:.3f}'.format((ck_sns + ck_amb + ck_nat) / 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--form_path_1', type=str, help='path to the file containing ratings from the first annotator',
                        required=True)
    parser.add_argument('--key_path_1', type=str, help='path to the key file containing sample details (e.g. sample provenance) for the first annotator',
                        required=True)
    parser.add_argument('--form_path_2', type=str, help='path to the file containing ratings from the second annotator',
                        required=True)
    parser.add_argument('--key_path_2', type=str, help='path to the key file containing sample details (e.g. sample provenance) for the second annotator',
                        required=True)
    args = parser.parse_args()

    read_tables(args.form_path_1, args.key_path_1, args.form_path_2, args.key_path_2)
