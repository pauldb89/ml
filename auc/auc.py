import math
import random
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(labels, predictions):
    """Function computing the area under the ROC curve.

    Params:
        labels - an array of {0, 1} values representing the true labels of the examples.
        predictions - an array of floats [0, 1] representing the outputs of a probabilistic binary classifier.
    """

    assert len(labels) == len(predictions), "labels and predictions must have the same size."

    assert set(labels) == set([0, 1]), "labels must contain only 0 or 1 and at least one 0 and 1."

    assert 0 <= min(predictions) and max(predictions) <= 1, "predictions must only contain values between [0, 1]."

    num_positive_labels = float(sum(labels))
    num_negative_labels = len(labels) - num_positive_labels

    updates = {}
    for prediction, label in zip(predictions, labels):
        true_update, false_update = updates.get(prediction, (0, 0))
        if label:
            true_update += 1
        else:
            false_update += 1
        updates[prediction] = (true_update, false_update)

    area = 0
    last_point = (1, 1)
    true_positives = num_positive_labels
    false_positives = num_negative_labels
    for _, (true_update, false_update) in sorted(updates.items()):
        true_positives -= true_update
        false_positives -= false_update

        new_point = (false_positives / num_negative_labels, true_positives / num_positive_labels)
        area += (last_point[0] - new_point[0]) * (last_point[1] + new_point[1]) / 2
        last_point = new_point

    return area


# TODO(pauldb): Move tests into their own directory.
# TODO(pauldb): Create an abstract base test case class.
class AucTestCase(unittest.TestCase):
    def test_mismatched_lengths(self):
        with self.assertRaises(AssertionError):
            auc([0, 1], [0.5])

    def test_invalid_labels(self):
        with self.assertRaises(AssertionError):
            auc([0, 2], [0.5, 0.5])

        with self.assertRaises(AssertionError):
            auc([0, 0], [0.5, 0.7])

        with self.assertRaises(AssertionError):
            auc([1, 1], [0.5, 0.7])

    def test_invalid_predictions(self):
        with self.assertRaises(AssertionError):
            auc([0, 1], [0, 1.1])

        with self.assertRaises(AssertionError):
            auc([0, 1], [-0.1, 1])

    def test_auc(self):
        self.assertEquals(1.0, auc([0, 1, 0, 1], [0, 1, 0, 1]))
        self.assertEquals(1.0, auc([0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9]))
        self.assertEquals(0.0, auc([1, 0, 1, 0], [0, 1, 0, 1]))
        self.assertEquals(0.0, auc([1, 0, 1, 0], [0.1, 0.9, 0.1, 0.9]))
        self.assertEquals(0.5, auc([0, 1, 0, 1], [1, 0, 0, 1]))

        for _ in range(5):
            for p in range(2, 6):
                size = int(math.pow(10, p))
                predictions = [random.uniform(0, 1) for _ in range(size)]
                labels = [random.randint(0, 1) for _ in range(size)]
                self.assertAlmostEquals(roc_auc_score(labels, predictions), auc(labels, predictions))

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
                self.assertAlmostEquals(roc_auc_score(labels, predictions), auc(labels, predictions))

    def test_shifted_auc(self):
        for _ in range(5):
            predictions = [random.uniform(0, 0.5) for _ in range(30)]
            shifted_predictions = [p + 0.5 for p in predictions]
            labels = [random.randint(0, 1) for _ in range(30)]

            self.assertEquals(auc(labels, predictions), auc(labels, shifted_predictions))


if __name__ == "__main__":
    unittest.main()
