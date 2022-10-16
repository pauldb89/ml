### Problem

Given N draws sampled uniformly from [0, 1], what is the distribution of the difference
between the nth and (n+1)th value after sorting?

### Code

An analytical derivation of the CDF can be found [here](https://math.stackexchange.com/questions/1180151/whats-the-probability-distribution-of-the-difference-between-two-consecutive-so).

This code estimates the PDF empirically. We then compare the empirically computed PDF
with the analytical solution as well as several similar candidates from the exponential
distribution family. To compare the distributions we:
- First plot the pdfs to visualize how similar they are.
- Compute the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measure (closer to 0 means more similar).
- Apply the [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) test which seeks to assert whether two distributions are different with a certain p-value.
