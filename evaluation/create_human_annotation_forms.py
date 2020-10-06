import re
import sys
import json
import random
import argparse
import pandas as pd


def prepare_mixed_corpus(natural_challenge_table_os,
                         natural_challenge_table_wmt,
                         synthetic_challenge_table_os,
                         synthetic_challenge_table_wmt,
                         sense_clusters_table,
                         csv_path):
    """ Writes the synthetic samples to a spreadsheet. """

    # Read in samples
    print('Reading-in the sub-sampled natural challenge sets ...')
    with open(natural_challenge_table_os, 'r', encoding='utf8') as nct_os:
        natural_challenge_samples_os = json.load(nct_os)
    with open(natural_challenge_table_wmt, 'r', encoding='utf8') as nct_wmt:
        natural_challenge_samples_wmt = json.load(nct_wmt)

    print('Reading-in the sub-sampled synthetic challenge sets ...')
    with open(synthetic_challenge_table_os, 'r', encoding='utf8') as sct_os:
        synthetic_challenge_samples_os = json.load(sct_os)
    with open(synthetic_challenge_table_wmt, 'r', encoding='utf8') as sct_wmt:
        synthetic_challenge_samples_wmt = json.load(sct_wmt)

    print('Reading-in known sense clusters ...')
    with open(sense_clusters_table, 'r', encoding='utf8') as sct:
        sense_clusters = json.load(sct)

    def _sense_to_string(sense_list):
        """ Helper function. """
        singular_forms = [sense_list[0]]
        all_senses = sense_list[1:]
        for sense_id, sense in enumerate(all_senses):
            is_plural = False
            for sf in singular_forms:
                if sf.lower() in sense.lower():
                    is_plural = True
                    break
            if not is_plural:
                singular_forms.append(sense)
        return ', '.join(singular_forms[:10])

    # Select natural rows
    unique_natural_rows = list()
    shared_natural_rows = list()
    for natural_challenge_samples, tag in [(natural_challenge_samples_os, 'os'),
                                           (natural_challenge_samples_wmt, 'wmt')]:
        all_natural_rows = list()
        for term in natural_challenge_samples.keys():
            for seed_cluster in natural_challenge_samples[term].keys():
                for adv_cluster in natural_challenge_samples[term][seed_cluster].keys():
                    for sample in natural_challenge_samples[term][seed_cluster][adv_cluster]:
                        # Avoid sentences with multiple instances of the homograph
                        if len(sample[6]) > 1:
                            continue
                        seed_sentence = sample[1].strip()
                        seed_sentence = re.sub(r'&apos;', '\'', seed_sentence)
                        seed_sentence = re.sub(r'&quot;', '\"', seed_sentence)
                        seed_sentence = re.sub(r'@-@', '-', seed_sentence)
                        # Look-up senses
                        true_senses = _sense_to_string(sorted(sample[14], reverse=False, key=lambda x: len(x)))
                        alt_senses = list(sense_clusters[term].keys())
                        alt_senses.remove(seed_cluster)
                        adv_senses_list = sense_clusters[term][random.choice(alt_senses)]['[SENSES]']
                        adv_senses = _sense_to_string(sorted(adv_senses_list, reverse=False, key=lambda x: len(x)))
                        sense_tuple = [(true_senses, True), (adv_senses, False)]
                        random.shuffle(sense_tuple)
                        all_natural_rows.append([seed_sentence, term, sense_tuple, 'natural', tag])
        random.shuffle(all_natural_rows)
        unique_natural_rows += all_natural_rows[:500]
        shared_natural_rows += all_natural_rows[500:600]

    # Select synthetic rows
    unique_synthetic_rows = list()
    shared_synthetic_rows = list()
    for synthetic_challenge_samples, tag in [(synthetic_challenge_samples_os, 'os'),
                                             (synthetic_challenge_samples_wmt, 'wmt')]:
        all_synthetic_rows = dict()
        seen_seeds = dict()
        for term in synthetic_challenge_samples.keys():
            for seed_cluster in synthetic_challenge_samples[term].keys():
                for adv_cluster in synthetic_challenge_samples[term][seed_cluster].keys():
                    for sample in synthetic_challenge_samples[term][seed_cluster][adv_cluster]:
                        # Avoid sentences with multiple instances of the homograph
                        if len(sample[6]) > 1:
                            continue
                        seed_sentence = sample[1].strip()
                        seed_sentence = re.sub(r'&apos;', '\'', seed_sentence)
                        seed_sentence = re.sub(r'&quot;', '\"', seed_sentence)
                        seed_sentence = re.sub(r'@-@', '-', seed_sentence)
                        # Avoid seeds with multiple term instances
                        if seen_seeds.get((seed_sentence, term), None) is not None:
                            continue
                        else:
                            seen_seeds[(seed_sentence, term)] = True
                        if all_synthetic_rows.get(seed_sentence, None) is None:
                            all_synthetic_rows[seed_sentence] = list()
                        adv_sentence = sample[0].strip()
                        adv_sentence = re.sub(r'&apos;', '\'', adv_sentence)
                        adv_sentence = re.sub(r'&quot;', '\"', adv_sentence)
                        adv_sentence = re.sub(r'@-@', '-', adv_sentence)
                        # Filter senses
                        true_senses = _sense_to_string(sorted(sample[14], reverse=False, key=lambda x: len(x)))
                        adv_senses = _sense_to_string(sorted(sample[15], reverse=False, key=lambda x: len(x)))
                        sense_tuple = [(true_senses, True), (adv_senses, False)]
                        random.shuffle(sense_tuple)
                        all_synthetic_rows[seed_sentence].append([adv_sentence, term, sense_tuple, 'synthetic', tag])

        all_seeds = list(all_synthetic_rows.keys())
        random.shuffle(all_seeds)
        sampled_seeds = all_seeds[:600]
        for ss in sampled_seeds[:500]:
            unique_synthetic_rows.append(random.choice(all_synthetic_rows[ss]))
        for ss in sampled_seeds[500:]:
            shared_synthetic_rows.append(random.choice(all_synthetic_rows[ss]))

    # Segment into sheets
    unique_rows = unique_natural_rows + unique_synthetic_rows
    random.shuffle(unique_rows)
    shared_rows = shared_natural_rows + shared_synthetic_rows
    random.shuffle(shared_rows)

    all_sheets = list()
    all_keys = list()
    curr_sheet = {'SENTENCE': [''],
                  'HOMOGRAPH': [''],
                  'SENSE 1': [''],
                  'SENSE 2': [''],
                  'WHICH SENSE IS CORRECT?': [''],
                  'IS THE HOMOGRAPH AMBIGUOUS?': [''],
                  'DOES THE SENTENCE MAKE SENSE?': ['']}
    curr_key = dict()
    for row in unique_rows:
        curr_sheet['SENTENCE'].append(row[0])
        curr_sheet['HOMOGRAPH'].append(row[1])
        curr_sheet['SENSE 1'].append(row[2][0][0]),
        curr_sheet['SENSE 2'].append(row[2][1][0])
        curr_sheet['WHICH SENSE IS CORRECT?'].append('')
        curr_sheet['IS THE HOMOGRAPH AMBIGUOUS?'].append('')
        curr_sheet['DOES THE SENTENCE MAKE SENSE?'].append('')
        curr_key[len(curr_sheet['SENTENCE']) - 1] = [row[4], row[3], row[2][0][1], row[2][1][1]]

        if len(curr_sheet['SENTENCE']) == 1000:
            for shared_row in shared_rows:
                curr_sheet['SENTENCE'].append(shared_row[0])
                curr_sheet['HOMOGRAPH'].append(shared_row[1])
                curr_sheet['SENSE 1'].append(shared_row[2][0][0]),
                curr_sheet['SENSE 2'].append(shared_row[2][1][0])
                curr_sheet['WHICH SENSE IS CORRECT?'].append('')
                curr_sheet['IS THE HOMOGRAPH AMBIGUOUS?'].append('')
                curr_sheet['DOES THE SENTENCE MAKE SENSE?'].append('')
                curr_key[len(curr_sheet['SENTENCE']) - 1] = \
                    [shared_row[4], shared_row[3], shared_row[2][0][1], shared_row[2][1][1]]

            all_sheets.append(pd.DataFrame(
                curr_sheet, columns=['SENTENCE', 'HOMOGRAPH', 'SENSE 1', 'SENSE 2',
                                     'WHICH SENSE IS CORRECT?', 'IS THE HOMOGRAPH AMBIGUOUS?',
                                     'DOES THE SENTENCE MAKE SENSE?']))
            curr_sheet = {'SENTENCE': [''],
                          'HOMOGRAPH': [''],
                          'SENSE 1': [''],
                          'SENSE 2': [''],
                          'WHICH SENSE IS CORRECT?': [''],
                          'IS THE HOMOGRAPH AMBIGUOUS?': [''],
                          'DOES THE SENTENCE MAKE SENSE?': ['']}
            all_keys.append(curr_key)
            curr_key = dict()

    print('Segmented corpus into {:d} sheets'.format(len(all_sheets)))

    # Export dataframe as CVS
    for sheet_id, sheet_df in enumerate(all_sheets):
        sheet_path = '{:s}_{:d}.csv'.format(csv_path, sheet_id)
        sheet_df.to_csv(path_or_buf=sheet_path, sep='\t')
        print('Saved sheet to: {:s}'.format(sheet_path))
    # Save keys as JSON
    for key_id, key_table in enumerate(all_keys):
        key_path = '{:s}_{:d}_key.json'.format(csv_path, key_id)
        with open(key_path, 'w', encoding='utf8') as json_file:
            json.dump(key_table, json_file, indent=3, sort_keys=True, ensure_ascii=False)
        print('Saved sense key to: {:s}'.format(key_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--natural_challenge_table_os', type=str, help='path to the file containing unperturbed OS18 sentences',
                        required=True)
    parser.add_argument('--natural_challenge_table_wmt', type=str, help='path to the file containing unperturbed WMT19 sentences',
                        required=True)
    parser.add_argument('--synthetic_challenge_table_os', type=int, help='path to the file containing adversarial OS18 samples',
                        required=True)
    parser.add_argument('--synthetic_challenge_table_wmt', type=str, help='path to the file containing adversarial WMT19 samples',
                        required=True)
    parser.add_argument('--sense_clusters_table', type=str, help='path to the file containing homograph sense clusters',
                        required=True)
    parser.add_argument('--csv_path', type=str, help='path to the output file',
                        required=True)
    args = parser.parse_args()

    prepare_mixed_corpus(args.natural_challenge_table_os, 
                         args.natural_challenge_table_wmt, 
                         args.synthetic_challenge_table_os, 
                         args.synthetic_challenge_table_wmt, 
                         args.sense_clusters_table, 
                         csv_path)
    
