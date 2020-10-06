import sys
import json
import argparse
import language_tool_python

import numpy as np


def check_grammar(adversarial_samples_path):
    """ Checks grammar preservation between natural seed sentences and adversarial samples that have been derived
    from them. """

    # Read in attractor phrase table
    print('Reading-in adversarial samples table ...')
    with open(adversarial_samples_path, 'r', encoding='utf8') as asp:
        adversarial_samples_table = json.load(asp)

    # Initialize trackers
    error_counts = list()
    seed_error_types = dict()
    adv_error_types = dict()

    # Initialize language tool
    tool = language_tool_python.LanguageTool('en-US')

    print('Evaluating samples ...')
    # Obtain scores based on seed sentence properties
    for term_id, term in enumerate(adversarial_samples_table.keys()):
        for seed_cluster in adversarial_samples_table[term].keys():
            for adv_cluster in adversarial_samples_table[term][seed_cluster].keys():
                for sample in adversarial_samples_table[term][seed_cluster][adv_cluster]:
                    seed_sentence = sample[1].strip()
                    adv_sample = sample[0].strip()

                    seed_matches = tool.check(seed_sentence)
                    adv_matches = tool.check(adv_sample)
                    num_seed_matches = 0
                    num_adv_matches = 0
                    for sm in seed_matches:
                        if sm.ruleId not in \
                                ['UPPERCASE_SENTENCE_START', 'PROFANITY', 'COMMA_PARENTHESIS_WHITESPACE']:
                            num_seed_matches += 1
                    for am in adv_matches:
                        if am.ruleId not in \
                                ['UPPERCASE_SENTENCE_START', 'PROFANITY', 'COMMA_PARENTHESIS_WHITESPACE']:
                            num_adv_matches += 1
                    error_counts.append((num_seed_matches, num_adv_matches))

                    for sm in seed_matches:
                        sm_match_type = sm.ruleId
                        if seed_error_types.get(sm_match_type, None) is None:
                            seed_error_types[sm_match_type] = 1
                        else:
                            seed_error_types[sm_match_type] += 1

                    for am in adv_matches:
                        am_match_type = am.ruleId
                        if adv_error_types.get(am_match_type, None) is None:
                            adv_error_types[am_match_type] = 1
                        else:
                            adv_error_types[am_match_type] += 1

                    if len(error_counts) % 1000 == 0 and len(error_counts) > 0:
                        print('Seen {:d} samples'.format(len(error_counts)))

    print('Seen {:d} samples'.format(len(error_counts)))
    # Report
    equal_errors = 0
    more_seed = 0
    more_adv = 0

    for ec in error_counts:
        if ec[0] == ec[1]:
            equal_errors += 1
        elif ec[0] > ec[1]:
            more_seed += 1
        else:
            more_adv += 1

    seed_mean_errors = np.mean([ec[0] for ec in error_counts])
    adv_mean_errors = np.mean([ec[1] for ec in error_counts])

    print('Number of samples with equal number of errors in seed and adv: {:d}'.format(equal_errors))
    print('Number of samples with more errors in the seed sentence: {:d}'.format(more_seed))
    print('Number of samples with more errors in the adversarial sample: {:d}'.format(more_adv))
    print('Mean number of errors in seed sentences: {:.4f}'.format(seed_mean_errors))
    print('Mean number of errors in adversarial samples: {:.4f}'.format(adv_mean_errors))
    print('=' * 20)
    print('ERROR TYPE COUNTS (seed | adv)')
    all_error_types = list(set(list(seed_error_types.keys()) + list(adv_error_types.keys())))
    for et in all_error_types:
        stc = seed_error_types.get(et, 0)
        atc = adv_error_types.get(et, 0)
        print('{:s} : {:d} | {:d}'.format(et, stc, atc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_samples_path', type=str, help='path to the file containing the generated adversarial samples',
                        required=True)
    args = parser.parse_args()

    check_grammar(args.adversarial_samples_path)

