import torch

import torch.nn.functional as F


class LogisticMixture:
    def __init__(self, num_mixtures: int, params: torch.Tensor, eps: float = 1e-5):
        self.num_mixtures = num_mixtures
        self.eps = eps

        batch_size, _, height, width = params.size()

        self.mixture_logits = params[:, :num_mixtures, :, :]

        self.means = params[:, num_mixtures:4*num_mixtures, :, :].view(batch_size, num_mixtures, 3, height, width)
        log_stds = params[:, 4*num_mixtures:7*num_mixtures, :, :].view(batch_size, num_mixtures, 3, height, width)
        self.stds = torch.exp(log_stds)
        self.coeffs = params[:, 7*num_mixtures:, :, :].view(batch_size, num_mixtures, 3, height, width)

    def log_p(self, x: torch.Tensor) -> torch.Tensor:
        # Rescale input to [-1, 1].
        x = (2 * x) - 1

        x = torch.unsqueeze(x, dim=1).expand(-1, self.num_mixtures, -1, -1, -1)

        means = []
        next_coeff_idx = 0
        for color_idx in range(3):
            mean = self.means[:, :, color_idx, :, :]
            for prev_color_idx in range(color_idx):
                mean += x[:, :, prev_color_idx, :, :] * self.coeffs[:, :, next_coeff_idx, :, :]
                next_coeff_idx += 1
            means.append(mean)

        means = torch.stack(means, dim=2)

        x_max = (x + 1 / 255. - means) / self.stds
        x_min = (x - 1 / 255. - means) / self.stds
        cdf_max = F.sigmoid(x_max)
        cdf_min = F.sigmoid(x_min)
        cdf_delta = cdf_max - cdf_min

        # Here we apply the following equivalence for numerical stability:
        # log sigmoid(x) = x - softplus(x)
        # Proof:
        # By definition: softplus(x) = log(1 + exp(x))
        # exp(rhs) = exp(x - softplus(x))
        #          = exp(x) / (1 + exp(x))
        #          = 1 / (((exp(0) + exp(x)) / exp(x)))
        #          = 1 / ((exp(0) / exp(x) + exp(x) / exp(x))
        #          = 1 / (1 + exp(-x)) = sigmoid(x) = exp(lhs)
        # log_cdf_max is only used when x falls in the leftmost bin and x_min should be mapped to -inf.
        # Since sigmoid(-inf) = 0 the probability mass for the leftmost bin is determined only by sigmoid(x_max).
        log_cdf_max = x_max - F.softplus(x_max)

        # Here we apply the following numerical equivalence for numerical stability:
        # log(1 - sigmoid(x)) = -softplus(x)
        # Proof:
        # log(1 - sigmoid(x)) = log(1 - 1 / (1 + exp(-x)))
        #                     = log((1 + exp(-x) - 1) / (1 + exp(-x)))
        #                     = log(exp(-x) / (1 + exp(-x)))
        # By multiplying the numerator and denominator by exp(x) we get:
        # lhs = log((exp(-x) * exp(x)) / (exp(x) * (1 - exp(-x)))
        #     = log( 1 / (exp(x) + 1))
        #     = -log(1 + exp(x)) = -softplus(x)
        # log_one_cdf_min is only used when x falls in the rightmost bin and x_max should be mapped to inf.
        # Since sigmoid(inf) = 1, the probability mass for the rightmost bin is 1 - sigmoid(x_min).
        log_one_cdf_min = -F.softplus(x_min)

        # TODO(pauldb): Either clamp the log or maybe add the last check numerical stability check that I don't
        # understand.
        log_per_mixture_probs = torch.where(
            x < -0.999,
            log_cdf_max,
            torch.where(x > 0.99, log_one_cdf_min, torch.log(cdf_delta))
        )

        log_probs = F.log_softmax(self.mixture_logits) + torch.sum(log_per_mixture_probs, dim=2)

        return torch.logsumexp(log_probs, dim=1)


