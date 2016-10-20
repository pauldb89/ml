from multiprocessing import Pool
from optparse import OptionParser
import os
import time

import tensorflow as tf


def process_file(args):
    input_filename, output_filename, vocab, word_clusters, ngram_size = args

    start_time = time.time()
    print 'Starting to process', input_filename

    with tf.RecordWriter(output_filename) as writer:
        for line in open(input_filename):
            sentence = ["<s>"] * (ngram_size - 1) + line.strip().split() + ["</s>"]
            word_ids = [vocab[word] for word in sentence]

            for i in range(ngram_size - 1, len(word_ids)):
                context = word_ids[i - ngram_size + 1: i]
                cluster = word_clusters[sentence[i]]
                next_word = word_ids[i]

                example = tf.train.Example(features=tf.train.Features(feature={
                    'context': tf.train.Feature(int64_list=tf.train.Int64List(values=context)),
                    'cluster': tf.train.Feature(int64_list=tf.train.Int64List(values=[cluster])),
                    'next_word': tf.train.Feature(int64_list=tf.train.Int64List(values=[next_word])),
                }))

                writer.write(example.SerializeToString())

    print 'Generating', output_filename, 'took', time.time() - start_time, 'seconds'


def main():
    parser = OptionParser()
    parser.add_option("--input_dir", dest="input_dir", help="Directory containing raw data.")
    parser.add_option("--output_dir", dest="output_dir", help="Directory where the training data is written")
    parser.add_option("--clusters_file", dest="clusters_file",
                      help="File containing the Brown clusters for the output layer factorization")
    parser.add_option("--ngram_size", dest="ngram_size", default=5, help="Language model window")
    options, _ = parser.parse_args()

    vocab = {"<s>": 0, "</s>": 1}
    word_clusters = {"<s>": 0, "</s>": 0}
    num_clusters = 1
    last_cluster = None
    for line in open(options.clusters_file):
        cluster, word, _ = line.strip().split()

        if cluster != last_cluster:
            last_cluster = cluster
            num_clusters += 1

        word_clusters[word] = num_clusters - 1
        vocab[word] = len(vocab)

    payloads = []
    for filename in os.listdir(input_dir):
        payloads.append((
            os.path.join(input_dir, filename), os.path.join(output_dir, filename), vocab, word_clusters, ngram_size))

    process_file(payloads[0])
    return

    pool = Pool(processes=4)
    pool.imap_async(process_file, payloads).get()


if __name__ == "__main__":
    main()
