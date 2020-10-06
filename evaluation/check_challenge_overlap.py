import sys
import json
import argparse

def check_overlap(set1_path, set2_path, l1):
    """ Checks challenge set overlap. """

    # Read-in sets
    print('Reading-in adversarial samples table ...')
    with open(set1_path, 'r', encoding='utf8') as s1p:
        set1 = json.load(s1p)
    with open(set2_path, 'r', encoding='utf8') as s2p:
        set2 = json.load(s2p)

    overlap = 0

    for term in set1.keys():
        if term not in set2.keys():
            continue
        s1_sents = list()
        s2_sents = list()

        for seed_cluster in set1[term].keys():
            for adv_cluster in set1[term][seed_cluster].keys():
                for seed_sentence in set1[term][seed_cluster][adv_cluster].keys():
                    s1_sents.append(seed_sentence)

        for seed_cluster in set2[term].keys():
            for adv_cluster in set2[term][seed_cluster].keys():
                for seed_sentence in set2[term][seed_cluster][adv_cluster].keys():
                    s2_sents.append(seed_sentence)

        overlap += len(list(set(s1_sents) & set(s2_sents)))

    print('Overlap: {:d} sentences ({:.2f}%)'.format(overlap, (overlap / l1) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path1', type=str, help='path to the first challenge set',
                        required=True)
    parser.add_argument('--corpus_path2', type=str, help='path to the second challenge set',
                        required=True)
    parser.add_argument('--corpus_size', type=int, help='challenge set size (has to be equal for both)',
                        required=True)
    args = parser.parse_args()

    check_overlap(sys.argv[1], sys.argv[2], int(sys.argv[3]))
