import re
import string
import langid
import argparse


def preprocess_bitext(src_path, tgt_path, src_lang, tgt_lang, max_len, max_len_ratio):
    """ Preprocesses the given bitext by removing empty lines, sentence pairs in incorrect languages,
    sentences above a the specified length threshold, and sentence pair exceeding the specified length ratio. """

    # Generate output paths
    src_out_path = '.'.join(src_path.split('.')[:-1]) + '.clean.{:s}'.format(src_lang)
    tgt_out_path = '.'.join(tgt_path.split('.')[:-1]) + '.clean.{:s}'.format(tgt_lang)

    # Open aligned corpora
    tgt_text = open(tgt_path, 'rb')

    print('Cleaning corpora ...')
    lines_kept = 0

    with open(src_path, 'rb') as src_text:
        with open(src_out_path, 'wb') as src_in:
            with open(tgt_out_path, 'wb') as tgt_in:
                for line_id, orig_src_line in enumerate(src_text):
                    orig_tgt_line = tgt_text.readline()

                    try:
                        str_src_line = orig_src_line.decode('utf-8')
                        str_tgt_line = orig_tgt_line.decode('utf-8')
                    except UnicodeDecodeError:
                        continue

                    # Remove punctuation
                    src_line = re.sub(r' +', ' ', str_src_line.strip().translate(pct_stripper))
                    tgt_line = re.sub(r' +', ' ', str_tgt_line.strip().translate(pct_stripper))
                    # NOTE: Lines which only contain whitespaces are not removed! This should be fixed.
                    # Keep if not empty
                    if len(src_line) > 0 and len(tgt_line) > 0:
                        # Keep if correct languages
                        if langid.classify(src_line)[0] == src_lang and langid.classify(tgt_line)[0] == tgt_lang:
                            # Tokenize
                            src_len = len(src_line.split(' '))
                            tgt_len = len(tgt_line.split(' '))
                            # Keep if below length threshold
                            if src_len <= max_len and tgt_len <= max_len:
                                # Keep if below length ratio
                                if max(src_len, tgt_len) / min(src_len, tgt_len) <= max_len_ratio:
                                    src_in.write(orig_src_line)
                                    tgt_in.write(orig_tgt_line)
                                    lines_kept += 1
                        # Report occasionally
                        if line_id > 0 and line_id % 100000 == 0:
                            print('Processed {:d} sentence pairs | Kept {:d}'.format(line_id, lines_kept))

    # Close open file objects
    tgt_text.close()

    print('-' * 20)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help='path to the source side of the parallel corpus to be cleaned',
                        required=True)
    parser.add_argument('--tgt_path', type=str, help='path to the target side of the parallel corpus to be cleaned',
                        required=True)
    parser.add_argument('--src_lang', type=str, help='denotes the source language, e.g. \'en\' for English',
                        required=True)
    parser.add_argument('--tgt_lang', type=str, help='denotes the target language, e.g. \'de\' for German',
                        required=True)
    parser.add_argument('--max_len', type=int, default=300,
                        help='threshold for the maximum allowed sentence length')
    parser.add_argument('--max_len_ratio', type=int, default=2.0,
                        help='threshold for maximum allowed sentence length ratio')
    args = parser.parse_args()

    # Initialize the processing pipeline (for op-level comments)
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Restrict set of languages for language identification
    langid.set_languages([args.src_lang, args.tgt_lang])

    preprocess_bitext(args.src_path, args.tgt_path, args.src_lang, args.tgt_lang, args.max_len, args.max_len_ratio)
