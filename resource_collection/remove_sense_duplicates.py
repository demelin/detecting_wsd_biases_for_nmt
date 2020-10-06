import sys
import json
import argparse


def _deduplicate_senses(bn_senses_path):
    """ Main function. """
    # Load scraped BabelNet sense clusters
    with open(bn_senses_path, 'r', encoding='utf8') as in_fo:
        sense_map = json.load(in_fo)

    print('=' * 20)
    print('De-duplicating senses ...')
    for term in sense_map.keys():
        for cluster in sense_map[term].keys():
            cluster_senses = [sns for sns in sense_map[term][cluster]['[SENSES]']]
            cluster_senses = list(set(cluster_senses))
            sense_map[term][cluster]['[SENSES]'] = cluster_senses
    print('Done!')

    # Dump to JSON
    output_path = bn_senses_path[:-5] + '_dedup.json'
    with open(output_path, 'w', encoding='utf8') as out_fo:
        json.dump(sense_map, out_fo, indent=3, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--senses_path', type=str, help='path to the scraped BabelNet senses',
                        required=True)
    args = parser.parse_args()

    _deduplicate_senses(args.senses_path)

