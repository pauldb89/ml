from optparse import OptionParser
import os
import Queue
from treading import Queue

import tensorflow as tf


def read_file(args)
    filename_queue, example_queue, vocab, word_clusters, ngram_size = args

    while True:
        filename = filename_queue.get(block=True)

        num_examples = 0
        for line in open(filename):
            sentence = ["<s>"] * (ngram_size - 1) + line.strip().split()
            word_ids = [vocab[word] for word in sentence]

            for i in range(ngram_size, len(word_ids)):
                num_examples += 1
                example = {
                    'context': word_ids[i - ngram_size + 1: i],
                    'word': word_ids[i],
                    'cluster': word_clusters[word_ids[i]],
                }
                example_queue.put(example, block=True)

        filename_queue.put(filename)

        print 'Done reading', num_examples, 'from', filename


def create_example_generator(input_dir, vocab, word_clusters, ngram_size, batch_size, num_threads):
    filename_queue = Queue.Queue()
    for filename in os.listdir(input_dir):
        filename_queue.put(filename)

    example_queue = Queue.Queue(maxsize=1000)
    for thread_num in int(num_threads):
        thread.start_new_thread(read_file, (filename_queue, example_queue, vocab, word_clusters, ngram_size))

    while True:
        batch = []
        for idx in range(batch_size):
            example = example_queue.get(block=True)
            #TODO(pauldb): Merge examples into a batch of training examples.

        yield batch


def build_model(vocab, embedding_size, ngram_size, stddev=1e-4):
    embeddings = tf.get_variable(
        name="embeddings",
        shape=[len(vocab), embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=stddev))
    context = tf.placeholder(tf.int64, shape=[None, ngram_size - 1], name="context")

    context_embeddings = tf.nn.embedding_lookup(embeddings, context)

    for i in range(ngram_size - 1):
        context_matrix = tf.get_variable(
            name="context_matrix_%d" % i,
            shape=[embedding_size, embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=stddev))

    # hidden_layer = tf.add_n([


def main():
    parser = OptionParser()
    parser.add_option("--training_dir", dest="training_dir", help="Directory containing training data")
    parser.add_option("--test_dir", dest="test_dir", help="Directory containing test data")
    parser.add_option("--clusters_file", dest="clusters_file",
                      help="File containing the Brown clustering for the output layer factorization")
    parser.add_option("--num_threads", dest="num_threads", default=1, help="Number of reader threads")
    parser.add_option("--ngram_size", dest="ngram_size", default=5, help="N-gram size scored by the model")
    parser.add_option("--embedding_size", dest="embedding_size", default=30, help="Size of word embeddings")
    parser.add_option("--batch_size", dest="batch_size", default=100, help="Training minibatch size")
    options, _ = parser.parse_args()

    num_threads = int(options.num_threads)
    ngram_size = int(options.ngram_size)
    embedding_size = int(options.embedding_size)
    batch_size = int(options.batch_size)

    vocab = {"<s>": 0}
    word_clusters = {}
    last_cluster = None
    num_clusters = 0
    for line in open(options.clusters_file):
        cluster, word, frequency = line.strip().split()

        if cluster != last_cluster:
            num_clusters += 1
            last_cluster = cluster

        word_id = len(vocab)
        vocab[word] = word_id
        word_clusters[word_id] = num_clusters - 1

    training_examples = create_example_generator(
        options.training_dir, vocab, word_clusters, ngram_size, batch_size, num_threads)
    # TODO(pauldb): Hard code the actual size of the test set.
    test_examples = create_example_generator(
        options.test_dir, vocab, word_clusters, ngram_size, batch_size=10000, num_threads=1)

    with tf.variable_scope("nlm") as scope:
        training_logits = build_model(vocab, embedding_size, ngram_size)
        scope.reuse_variables()
        test_logits = build_model(vocab, embedding_size, ngram_size)


if __name__ == "__main__":
    main()
