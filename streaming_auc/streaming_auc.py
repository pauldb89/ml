import math
import random
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score


class StreamingAUC(object):
    """Class for computing the area under the ROC curve for a probabilistic classifier
       in streaming fashion."""

    def __init__(self, num_thresholds=1000):
        self.num_thresholds = num_thresholds
        self.positive_buckets = [0] * num_thresholds
        self.negative_buckets = [0] * num_thresholds
        self.bucket_size = 1.0 / num_thresholds

    def update(self, labels, predictions):
        assert len(labels) == len(predictions), "labels and predictions must have the same size."

        assert set(labels).issubset(set([0, 1])), "labels must contain only 0 and 1."

        assert 0 <= min(predictions) and max(predictions) <= 1, "predictions must only contain values between [0, 1]."

        for prediction, label in zip(predictions, labels):
            # Predictions equal to 1 are allocated in the last bucket.
            bucket = min(int(prediction / self.bucket_size), self.num_thresholds - 1)
            if label:
                self.positive_buckets[bucket] += 1
            else:
                self.negative_buckets[bucket] += 1

    def compute(self):
        num_positive_labels = float(sum(self.positive_buckets))
        num_negative_labels = float(sum(self.negative_buckets))

        assert num_positive_labels > 0 and num_negative_labels > 0, "At least one positive and negative label is needed."

        last_point = (1, 1)
        area = 0
        true_positives = num_positive_labels
        false_positives = num_negative_labels
        for i in range(self.num_thresholds):
            true_positives -= self.positive_buckets[i]
            false_positives -= self.negative_buckets[i]

            new_point = (false_positives / num_negative_labels, true_positives / num_positive_labels)
            area += (last_point[0] - new_point[0]) * (last_point[1] + new_point[1]) / 2
            last_point = new_point

        return area


# TODO(pauldb): Move tests into their own directory.
# TODO(pauldb): Create an abstract base test case class.
class StreamingAUCTestCase(unittest.TestCase):
    def setUp(self):
        self.auc = StreamingAUC()

    def test_update_mismatched_lengths(self):
        with self.assertRaises(AssertionError):
            self.auc.update([0, 1], [0.5])

    def test_invalid_labels(self):
        with self.assertRaises(AssertionError):
            self.auc.update([0, 2], [0.5, 0.7])

        with self.assertRaises(AssertionError):
            self.auc.update([-1, 1], [0.5, 0.7])

        self.auc.update([0], [0.5])
        self.auc.update([1], [0.7])
        self.auc.update([0, 1], [0.4, 0.8])

    def test_invalid_predictions(self):
        with self.assertRaises(AssertionError):
            self.auc.update([0], [-0.1])

        with self.assertRaises(AssertionError):
            self.auc.update([1], [1.2])

        self.auc.update([0, 1], [0.2, 0.8])

    def test_compute_must_have_both_labels(self):
        with self.assertRaises(AssertionError):
            self.auc.compute()

        self.auc.update([0], [0.1])
        with self.assertRaises(AssertionError):
            self.auc.compute()

        self.auc.update([1], [0.7])
        self.auc.compute()

    def test_single_batch_auc(self):
        auc = StreamingAUC()
        auc.update([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEquals(1, auc.compute())

        auc = StreamingAUC()
        auc.update([0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9])
        self.assertEquals(1, auc.compute())

        auc = StreamingAUC()
        auc.update([1, 0, 1, 0], [0, 1, 0, 1])
        self.assertEquals(0, auc.compute())

        auc = StreamingAUC()
        auc.update([1, 0, 1, 0], [0.1, 0.9, 0.1, 0.9])
        self.assertEquals(0, auc.compute())

        auc = StreamingAUC()
        auc.update([0, 1, 0, 1], [1, 0, 0, 1])
        self.assertEquals(0.5, auc.compute())

        for _ in range(5):
            for p in range(2, 6):
                size = int(math.pow(10, p))
                predictions = [random.uniform(0, 1) for _ in range(size)]
                labels = [random.randint(0, 1) for _ in range(size)]
                auc = StreamingAUC(num_thresholds=10000)
                auc.update(labels, predictions)
                self.assertAlmostEquals(roc_auc_score(labels, predictions), auc.compute(), places=3)

                predictions = []
                labels = []
                for i in range(2):
                    mean = random.uniform(0, 1)
                    stddev = random.uniform(0, 0.2)
                    predictions.extend([np.clip(random.normalvariate(mean, stddev), 0, 1) for _ in range(size)])
                    labels.extend([i for _ in range(size)])

                indices = random.sample(range(len(predictions)), len(predictions))
                predictions = [predictions[i] for i in indices]
                labels = [labels[i] for i in indices]

                auc = StreamingAUC(num_thresholds=10000)
                auc.update(labels, predictions)
                self.assertAlmostEquals(roc_auc_score(labels, predictions), auc.compute(), places=3)

    def test_streaming_auc(self):
        for _ in range(50):
            auc = StreamingAUC(num_thresholds=10000)

            all_predictions = []
            all_labels = []
            for _ in range(10):
                predictions = [random.uniform(0, 1) for _ in range(100)]
                labels = [random.randint(0, 1) for _ in range(100)]
                auc.update(labels, predictions)

                all_predictions.extend(predictions)
                all_labels.extend(labels)
                self.assertAlmostEquals(roc_auc_score(all_labels, all_predictions), auc.compute(), places=3)

    def test_shifted_auc(self):
        for _ in range(5):
            predictions = [random.uniform(0, 0.5) for _ in range(30)]
            shifted_predictions = [p + 0.5 for p in predictions]
            labels = [random.randint(0, 1) for _ in range(30)]

            auc = StreamingAUC()
            streaming_auc = StreamingAUC()

            auc.update(labels, predictions)
            streaming_auc.update(labels, shifted_predictions)
            self.assertEquals(auc.compute(), streaming_auc.compute())


if __name__ == "__main__":
    unittest.main()
