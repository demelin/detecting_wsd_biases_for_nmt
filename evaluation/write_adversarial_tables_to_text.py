import sys
import json
import argparse


def adv_samples_to_text(adversarial_samples_path, src_lang, compress_true_samples=False):

    """ Converts the table of true and adversarial source samples to text files that can be passed to a
    translation model. """

    # Read in attractor phrase table
    print('Reading-in adversarial samples table ...')
    with open(adversarial_samples_path, 'r', encoding='utf8') as asp:
        adversarial_samples_table = json.load(asp)

    # Check data type
    has_forms = None
    while has_forms is None:
        try:
            test_key1 = list(adversarial_samples_table.keys())[0]
            test_key2 = list(adversarial_samples_table[test_key1].keys())[0]
            test_key3 = list(adversarial_samples_table[test_key1][test_key2].keys())[0]
            has_forms = type(adversarial_samples_table[test_key1][test_key2][test_key3]) == dict
        except IndexError:
            continue

    # Open destination files
    target_dir = '/'.join(adversarial_samples_path.split('/')[:-1])
    target_file_name = adversarial_samples_path.split('/')[-1][:-5]
    nmt_true_src_path = '{:s}/text_files/{:s}_true_sources.{:s}'.format(target_dir, target_file_name, src_lang)
    nmt_adv_src_path = '{:s}/text_files/{:s}_adversarial_sources.{:s}'.format(target_dir, target_file_name, src_lang)
    nmt_true_src = open(nmt_true_src_path, 'w', encoding='utf8')
    nmt_adv_src = open(nmt_adv_src_path, 'w', encoding='utf8')

    # Write generated adversarial samples to file and document file lines in the adversarial table
    true_line_count, adv_line_count = 0, 0
    unique_true_src = dict()
    for term in adversarial_samples_table.keys():
        if has_forms:
            for form in adversarial_samples_table[term].keys():
                for true_cluster in adversarial_samples_table[term][form].keys():
                    for adv_cluster in adversarial_samples_table[term][form][true_cluster].keys():
                        for entry in adversarial_samples_table[term][form][true_cluster][adv_cluster]:

                            # Write to adversarial file
                            nmt_adv_src.write(entry[0].strip() + '\n')
                            entry.append(('adversarial translation line ', adv_line_count))
                            adv_line_count += 1
                            # Maybe write to original file
                            if compress_true_samples:
                                if not unique_true_src.get(entry[1], None):
                                    nmt_true_src.write(entry[1].strip() + '\n')
                                    unique_true_src[entry[1]] = true_line_count
                                    true_line_count += 1
                                entry.append(('true translation line ', unique_true_src[entry[1]]))
                            else:
                                nmt_true_src.write(entry[1].strip() + '\n')
                                entry.append(('true translation line ', true_line_count))
                                true_line_count += 1
                                assert true_line_count == adv_line_count, 'Count mismatch!'

                    # Report occasionally
                    if adv_line_count > 0 and adv_line_count % 1000 == 0:
                        print('Wrote {:d} adversarial samples to the NMT source input file.'.format(adv_line_count))

        else:
            for true_cluster in adversarial_samples_table[term].keys():
                for adv_cluster in adversarial_samples_table[term][true_cluster].keys():
                    for entry in adversarial_samples_table[term][true_cluster][adv_cluster]:

                        # Write to adversarial file
                        nmt_adv_src.write(entry[0].strip() + '\n')
                        entry.append(('adversarial translation line ', adv_line_count))
                        adv_line_count += 1
                        if compress_true_samples:
                            # Maybe write to original file
                            if not unique_true_src.get(entry[1], None):
                                nmt_true_src.write(entry[1].strip() + '\n')
                                unique_true_src[entry[1]] = true_line_count
                                true_line_count += 1
                            entry.append(('true translation line ', unique_true_src[entry[1]]))
                        else:
                            nmt_true_src.write(entry[1].strip() + '\n')
                            entry.append(('true translation line ', true_line_count))
                            true_line_count += 1
                            assert true_line_count == adv_line_count, 'Count mismatch!'

                # Report occasionally
                if adv_line_count > 0 and adv_line_count % 1000 == 0:
                    print('Wrote {:d} adversarial samples to the NMT source input file.'.format(adv_line_count))

    # Write adversarial table augmented with translation line numbers to file
    updated_adversarial_samples_path = adversarial_samples_path[:-5] + '_with_translation_ids.json'
    with open(updated_adversarial_samples_path, 'w', encoding='utf8') as usp:
        json.dump(adversarial_samples_table, usp, indent=3, sort_keys=True, ensure_ascii=False)
    print('Wrote the updated adversarial samples table to {:s}'.format(updated_adversarial_samples_path))

    # Open destination files
    nmt_true_src.close()
    nmt_adv_src.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_samples_path', type=str, help='path to the JSON table containing samples to be written to plain text file',
                        required=True)
    parser.add_argument('--src_lang', type=str, help='language identifier to be appended to file',
                        required=True)
    args = parser.parse_args()

    adv_samples_to_text(args.adversarial_samples_path, args.src_lang)
