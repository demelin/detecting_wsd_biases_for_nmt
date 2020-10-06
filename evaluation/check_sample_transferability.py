import os
import sys
import json
import argparse


def check_intersection(label1, label2, label3, table_path1, table_path2, table_path3):
    """ Evaluates how many samples occur in both of the specified tables and displays some relevant statistics. """

    if label3 == 'None':
        label3 = None
        table_path3 = None

    # Read-in tables
    with open(table_path1, 'r', encoding='utf8') as tp1:
        sample_table1 = json.load(tp1)
    with open(table_path2, 'r', encoding='utf8') as tp2:
        sample_table2 = json.load(tp2)
    sample_table3 = None
    if table_path3 is not None:
        with open(table_path3, 'r', encoding='utf8') as tp3:
            sample_table3 = json.load(tp3)

    # Initialize intersection table
    intersection_table = dict()
    seed_total_count = 0
    seed_intersection_count = 0
    sample_intersection_count = 0
    num_table1_samples = 0
    num_table2_samples = 0
    num_table3_samples = 0

    # Check intersection
    if 'true_samples' not in label1:
        for term in sample_table1.keys():
            for seed_cluster in sample_table1[term].keys():
                for adv_cluster in sample_table1[term][seed_cluster].keys():
                    for seed_sentence in sample_table1[term][seed_cluster][adv_cluster].keys():
                        sample_tuples1 = list()
                        sample_tuples2 = list()
                        sample_tuples3 = list()
                        seed_total_count += 1
                        for sample in sample_table1[term][seed_cluster][adv_cluster][seed_sentence]:

                            # Only consider samples derived from correctly translated seeds
                            if 'attractor' not in table_path1:
                                if sample[20][0] != 'not_flipped':
                                    continue

                            sample_tuples1.append((sample[0], sample[5], sample[6], sample[11][0]))
                            num_table1_samples += 1
                        try:
                            for sample in sample_table2[term][seed_cluster][adv_cluster][seed_sentence]:

                                # Only consider samples derived from correctly translated seeds
                                if 'attractor' not in table_path2:
                                    if sample[20][0] != 'not_flipped':
                                        continue

                                sample_tuples2.append((sample[0], sample[5], sample[6], sample[11][0]))
                        except KeyError:
                            continue
                        if sample_table3 is None:
                            if len(sample_tuples1) > 0 and len(sample_tuples2) > 0:
                                seed_intersection_count += 1
                                for tpl1_index, tpl1 in enumerate(sample_tuples1):
                                    try:
                                        tpl2_index = sample_tuples2.index(tpl1)
                                    except ValueError:
                                        tpl2_index = None
                                    if tpl2_index is not None:
                                        # Increment count
                                        sample_intersection_count += 1
                                        # Extend table
                                        if intersection_table.get(term, None) is None:
                                            intersection_table[term] = dict()
                                        if intersection_table[term].get(seed_cluster, None) is None:
                                            intersection_table[term][seed_cluster] = dict()
                                        if intersection_table[term][seed_cluster].get(adv_cluster, None) is None:
                                            intersection_table[term][seed_cluster][adv_cluster] = dict()
                                        if intersection_table[term][seed_cluster][adv_cluster]\
                                                .get(seed_sentence, None) is None:
                                            intersection_table[term][seed_cluster][adv_cluster][seed_sentence] = list()
                                        intersection_table[term][seed_cluster][adv_cluster][seed_sentence].append(
                                            (label1,
                                             sample_table1[term][seed_cluster][adv_cluster][seed_sentence][tpl1_index],
                                             label2,
                                             sample_table2[term][seed_cluster][adv_cluster][seed_sentence][tpl2_index]))

                        else:
                            try:
                                for sample in sample_table3[term][seed_cluster][adv_cluster][seed_sentence]:

                                    # Only consider samples derived from correctly translated seeds
                                    if 'attractor' not in table_path3:
                                        if sample[20][0] != 'not_flipped':
                                            continue

                                    sample_tuples3.append((sample[0], sample[5], sample[6], sample[11][0]))
                            except KeyError:
                                continue
                            if len(sample_tuples1) > 0 and len(sample_tuples2) > 0 and len(sample_tuples3) > 0:
                                seed_intersection_count += 1
                                for tpl1_index, tpl1 in enumerate(sample_tuples1):
                                    try:
                                        tpl2_index = sample_tuples2.index(tpl1)
                                        tpl3_index = sample_tuples3.index(tpl1)
                                    except ValueError:
                                        tpl2_index = None
                                        tpl3_index = None
                                    if tpl2_index is not None and tpl3_index is not None:
                                        # Increment count
                                        sample_intersection_count += 1
                                        # Extend table
                                        if intersection_table.get(term, None) is None:
                                            intersection_table[term] = dict()
                                        if intersection_table[term].get(seed_cluster, None) is None:
                                            intersection_table[term][seed_cluster] = dict()
                                        if intersection_table[term][seed_cluster].get(adv_cluster, None) is None:
                                            intersection_table[term][seed_cluster][adv_cluster] = dict()
                                        if intersection_table[term][seed_cluster][adv_cluster]\
                                                .get(seed_sentence, None) is None:
                                            intersection_table[term][seed_cluster][adv_cluster][seed_sentence] = list()
                                        intersection_table[term][seed_cluster][adv_cluster][seed_sentence].append(
                                            (label1,
                                             sample_table1[term][seed_cluster][adv_cluster][seed_sentence][tpl1_index],
                                             label2,
                                             sample_table2[term][seed_cluster][adv_cluster][seed_sentence][tpl2_index],
                                             label3,
                                             sample_table3[term][seed_cluster][adv_cluster][seed_sentence][tpl3_index]))

        # Count number of samples in table2
        for term in sample_table2.keys():
            for seed_cluster in sample_table2[term].keys():
                for adv_cluster in sample_table2[term][seed_cluster].keys():
                    for seed_sentence in sample_table2[term][seed_cluster][adv_cluster].keys():
                        for sample in sample_table2[term][seed_cluster][adv_cluster][seed_sentence]:
                            if 'attractor' not in table_path2:
                                if sample[20][0] == 'not_flipped':
                                    num_table2_samples += 1
                            else:
                                num_table2_samples += 1

        if sample_table3 is not None:
            # Count number of samples in table3
            for term in sample_table3.keys():
                for seed_cluster in sample_table3[term].keys():
                    for adv_cluster in sample_table3[term][seed_cluster].keys():
                        for seed_sentence in sample_table3[term][seed_cluster][adv_cluster].keys():
                            for sample in sample_table3[term][seed_cluster][adv_cluster][seed_sentence]:
                                if 'attractor' not in table_path3:
                                    if sample[20][0] == 'not_flipped':
                                        num_table3_samples += 1
                                else:
                                    num_table3_samples += 1

        # Report
        print('-' * 20)
        print('Table {:s} contains {:d} samples'.format(label1, num_table1_samples))
        print('Table {:s} contains {:d} samples'.format(label2, num_table2_samples))
        if sample_table3 is not None:
            print('Table {:s} contains {:d} samples'.format(label3, num_table3_samples))
            jaccard_similarity = sample_intersection_count / (num_table1_samples + num_table2_samples +
                                                              num_table3_samples - sample_intersection_count)
        else:
            jaccard_similarity = \
                sample_intersection_count / (num_table1_samples + num_table2_samples - sample_intersection_count)

        print('Overlap / Jaccard similarity: {:d} | {:.4f}'.format(sample_intersection_count, jaccard_similarity))

    else:
        for term in sample_table1.keys():
            for seed_cluster in sample_table1[term].keys():
                seed_cluster_pool1 = list()
                for adv_cluster in sample_table1[term][seed_cluster].keys():
                    seed_cluster_pool1 += \
                        [s for s in list(sample_table1[term][seed_cluster][adv_cluster].keys())]
                seed_cluster_pool2 = list()
                seed_cluster_pool3 = list()
                try:
                    for adv_cluster in sample_table2[term][seed_cluster].keys():
                        seed_cluster_pool2 += \
                            [s for s in list(sample_table2[term][seed_cluster][adv_cluster].keys())]
                except KeyError:
                    continue
                if sample_table3 is None:
                    if len(seed_cluster_pool1) > 0 and len(seed_cluster_pool2) > 0:
                        seed_cluster_sents1 = list(set([tpl[0] for tpl in seed_cluster_pool1]))
                        seed_cluster_sents2 = list(set([tpl[0] for tpl in seed_cluster_pool2]))
                        for ss1_index, ss1 in enumerate(seed_cluster_sents1):
                            try:
                                ss2_index = seed_cluster_sents2.index(ss1)
                            except ValueError:
                                ss2_index = None
                            if ss2_index is not None:
                                # Increment count
                                seed_intersection_count += 1
                                # Extend table
                                if intersection_table.get(term, None) is None:
                                    intersection_table[term] = dict()
                                if intersection_table[term].get(seed_cluster, None) is None:
                                    intersection_table[term][seed_cluster] = list()
                                intersection_table[term][seed_cluster].append(ss1)
                else:
                    try:
                        for adv_cluster in sample_table3[term][seed_cluster].keys():
                            seed_cluster_pool3 += \
                                [s for s in list(sample_table3[term][seed_cluster][adv_cluster].keys())]
                    except KeyError:
                        continue
                    if len(seed_cluster_pool1) > 0 and len(seed_cluster_pool2) > 0 and len(seed_cluster_pool3) > 0:
                        seed_cluster_sents1 = list(set([tpl[0] for tpl in seed_cluster_pool1]))
                        seed_cluster_sents2 = list(set([tpl[0] for tpl in seed_cluster_pool2]))
                        seed_cluster_sents3 = list(set([tpl[0] for tpl in seed_cluster_pool3]))
                        for ss1_index, ss1 in enumerate(seed_cluster_sents1):
                            try:
                                ss2_index = seed_cluster_sents2.index(ss1)
                                ss3_index = seed_cluster_sents3.index(ss1)
                            except ValueError:
                                ss2_index = None
                                ss3_index = None
                            if ss2_index is not None and ss3_index is not None:
                                # Increment count
                                seed_intersection_count += 1
                                # Extend table
                                if intersection_table.get(term, None) is None:
                                    intersection_table[term] = dict()
                                if intersection_table[term].get(seed_cluster, None) is None:
                                    intersection_table[term][seed_cluster] = list()
                                intersection_table[term][seed_cluster].append(ss1)

        # Count number of samples in table1
        for term in sample_table1.keys():
            for seed_cluster in sample_table1[term].keys():
                for adv_cluster in sample_table1[term][seed_cluster].keys():
                    num_table1_samples += len(sample_table1[term][seed_cluster][adv_cluster].keys())

        # Count number of samples in table2
        for term in sample_table2.keys():
            for seed_cluster in sample_table2[term].keys():
                for adv_cluster in sample_table2[term][seed_cluster].keys():
                    num_table2_samples += len(sample_table2[term][seed_cluster][adv_cluster].keys())

        if sample_table3 is not None:
            # Count number of samples in table3
            for term in sample_table3.keys():
                for seed_cluster in sample_table3[term].keys():
                    for adv_cluster in sample_table3[term][seed_cluster].keys():
                        num_table3_samples += len(sample_table3[term][seed_cluster][adv_cluster].keys())

        # Report
        print('-' * 20)
        print('Table {:s} contains {:d} seeds'.format(label1, num_table1_samples))
        print('Table {:s} contains {:d} seeds'.format(label2, num_table2_samples))
        if sample_table3 is not None:
            print('Table {:s} contains {:d} seeds'.format(label3, num_table3_samples))
            jaccard_similarity = seed_intersection_count / (num_table1_samples + num_table2_samples +
                                                            num_table3_samples - seed_intersection_count)
        else:
            jaccard_similarity = \
                seed_intersection_count / (num_table1_samples + num_table2_samples - seed_intersection_count)

        print('Overlap / Jaccard similarity: {:d} | {:.4f}'.format(seed_intersection_count, jaccard_similarity))

    # Save
    # Open destination file
    intersection_dir = '/'.join(table_path1.split('/')[:-3]) + '/sample_intersections'
    if not os.path.isdir(intersection_dir):
        os.mkdir(intersection_dir)
    if sample_table3 is None:
        out_path = '{:s}/{:s}.AND.{:s}.json'.format(intersection_dir, label1, label2)
    else:
        out_path = '{:s}/{:s}.AND.{:s}.AND.{:s}.json'.format(intersection_dir, label1, label2, label3)

    with open(out_path, 'w', encoding='utf8') as op:
        json.dump(intersection_table, op, indent=3, sort_keys=True, ensure_ascii=False)
    print('Saved samples found in all tables to {:s}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label1', type=str, help='label associated with the first set of samples',
                        required=True)
    parser.add_argument('--label2', type=str, help='label associated with the second set of samples',
                        required=True)
    parser.add_argument('--label3', type=str, help='label associated with the third set of samples')
    parser.add_argument('--corpus_path1', type=str, help='path to the first set of samples used in similarity calculation',
                        required=True)
    parser.add_argument('--corpus_path2', type=str, help='path to the second set of samples used in similarity calculation',
                        required=True)
    parser.add_argument('--corpus_path3', type=str, help='path to the third set of samples used in similarity calculation')
    args = parser.parse_args()

    check_intersection(args.label1, args.label2, args.label3, args.corpus_path1, args.corpus_path2, args.corpus_path3)

