from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy
import numpy as np
import torch
import torch.nn.functional as F


def add_backward(grad: np.ndarray, source_tensors: List[Tensor]) -> Dict[Tensor, np.ndarray]:
    ret = {}
    for source_tensor in source_tensors:
        if source_tensor.requires_grad:
            ret[source_tensor] = grad

    return ret


def sum_backward(grad: np.ndarray, source_tensors: List[Tensor]) -> Dict[Tensor, np.ndarray]:
    assert len(source_tensors) == 1
    assert len(grad.shape) == 0

    source_tensor = source_tensors[0]
    return {source_tensor: np.full_like(source_tensor.value, fill_value=grad.item())}


def mat_mult_backward(grad: np.ndarray, source_tensors: List[Tensor]) -> Dict[Tensor, np.ndarray]:
    ret = {}
    if source_tensors[0].requires_grad:
        ret[source_tensors[0]] = np.matmul(grad, source_tensors[1].value.transpose())
    if source_tensors[1].requires_grad:
        ret[source_tensors[1]] = np.matmul(source_tensors[0].value.transpose(), grad)
    return ret


def mse_backward(grad: np.ndarray, source_tensors: List[Tensor]) -> Dict[Tensor, np.ndarray]:
    ret = {}
    if source_tensors[0].requires_grad:
        ret[source_tensors[0]] = 2 * (source_tensors[0].value - source_tensors[1].value) * grad

    if source_tensors[1].requires_grad:
        ret[source_tensors[1]] = -2 * (source_tensors[0].value - source_tensors[1].value) * grad

    return ret


def cross_entropy_backward(grad: np.ndarray, source_tensors: List[Tensor]) -> Dict[Tensor, np.ndarray]:
    ret = {}

    logits = source_tensors[0].value
    target = source_tensors[1].value
    batch_size, num_values = logits.shape

    if source_tensors[0].requires_grad:
        value = np.zeros(shape=(batch_size, num_values))
        value[np.arange(batch_size), target] = 1
        value -= np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        ret[source_tensors[0]] = -value * grad

    assert not source_tensors[1].requires_grad

    return ret


@dataclass
class Tensor:
    value: np.ndarray
    sources: List[Tensor]

    requires_grad: bool
    backprop: Optional[Callable[[np.ndarray, List[Tensor]], Dict[Tensor, numpy.ndarray]]]

    grad: Optional[np.ndarray] = None

    def __hash__(self) -> int:
        return id(self)

    @classmethod
    def topological_sort(cls, tensor: Tensor, order: List[Tensor], visited: Set[Tensor]) -> None:
        visited.add(tensor)
        for source_tensor in tensor.sources:
            if source_tensor.requires_grad and source_tensor not in visited:
                cls.topological_sort(source_tensor, order, visited)
        order.append(tensor)

    def backward(self) -> None:
        assert self.requires_grad, "The tensor must require gradient"
        assert len(self.value.shape) == 0, "Backward can only be called on a scalar"

        self.grad = np.array(1.0)
        sorted_tensors = []
        self.topological_sort(self, sorted_tensors, visited=set())

        for tensor in reversed(sorted_tensors):
            if not tensor.sources:
                continue

            for source_tensor in tensor.sources:
                if source_tensor.requires_grad and source_tensor.grad is None:
                    source_tensor.grad = np.zeros_like(source_tensor.value)

            source_grads = tensor.backprop(tensor.grad, tensor.sources)
            for source_tensor in tensor.sources:
                assert (source_tensor in source_grads) == source_tensor.requires_grad
                if source_tensor.requires_grad:
                    source_grad = source_grads[source_tensor]
                    assert source_tensor.grad.shape == source_grad.shape
                    source_tensor.grad += source_grad


def add(left: Tensor, right: Tensor) -> Tensor:
    assert left.value.shape == right.value.shape

    return Tensor(
        value=left.value + right.value,
        requires_grad=left.requires_grad or right.requires_grad,
        sources=[left, right],
        backprop=add_backward,
    )


def sum(tensor: Tensor) -> Tensor:
    return Tensor(
        value=np.sum(tensor.value),
        requires_grad=tensor.requires_grad,
        sources=[tensor],
        backprop=sum_backward,
    )


def matmul(left: Tensor, right: Tensor) -> Tensor:
    return Tensor(
        value=np.matmul(left.value, right.value),
        requires_grad=left.requires_grad or right.requires_grad,
        sources=[left, right],
        backprop=mat_mult_backward,
    )


def mse_loss(predicted: Tensor, target: Tensor) -> Tensor:
    return Tensor(
        value=np.sum(np.square(predicted.value - target.value)),
        requires_grad=predicted.requires_grad or target.requires_grad,
        sources=[predicted, target],
        backprop=mse_backward,
    )


def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    return Tensor(
        value=-np.sum(np.choose(target.value, logits.value.T) - np.log(np.sum(np.exp(logits.value), axis=-1))),
        requires_grad=logits.requires_grad or target.requires_grad,
        sources=[logits, target],
        backprop=cross_entropy_backward,
    )


def tensor(value: np.ndarray, requires_grad: bool = False):
    return Tensor(
        value=value,
        requires_grad=requires_grad,
        sources=[],
        backprop=None,
        grad=None,
    )


def test_one():
    a = tensor(np.full(shape=(2, 3), fill_value=1), requires_grad=True)
    b = tensor(np.full(shape=(2, 3), fill_value=5), requires_grad=True)
    c = add(a, b)
    d = sum(c)

    ta = torch.full(size=(2, 3), fill_value=1.0, requires_grad=True)
    tb = torch.full(size=(2, 3), fill_value=5.0, requires_grad=True)
    tc = ta + tb
    td = torch.sum(tc)

    np.testing.assert_almost_equal(d.value, td.detach().numpy(), decimal=4)

    expected_matches = {a: ta, b: tb, c: tc, d: td}
    for expected_tensor in expected_matches.values():
        expected_tensor.retain_grad()

    d.backward()
    td.backward()

    for actual_tensor, expected_tensor in expected_matches.items():
        np.testing.assert_almost_equal(actual_tensor.grad, expected_tensor.grad, decimal=4)


def test_two():
    example = tensor(np.random.rand(2, 3), requires_grad=False)
    weights_1 = tensor(np.random.rand(3, 5), requires_grad=True)

    hidden = matmul(example, weights_1)

    weights_2 = tensor(np.random.rand(5, 5), requires_grad=True)

    predicted = matmul(hidden, weights_2)
    mse_target = tensor(np.random.rand(2, 5), requires_grad=False)
    xent_target = tensor(np.random.randint(0, 4, size=(2,)), requires_grad=False)

    loss_1 = mse_loss(predicted, mse_target)
    loss_2 = cross_entropy(predicted, xent_target)
    loss = add(loss_1, loss_2)

    t_example = torch.tensor(example.value, requires_grad=False, dtype=torch.float)
    t_weights_1 = torch.tensor(weights_1.value, requires_grad=True, dtype=torch.float)
    t_weights_2 = torch.tensor(weights_2.value, requires_grad=True, dtype=torch.float)

    t_hidden = torch.matmul(t_example, t_weights_1)
    t_predicted = torch.matmul(t_hidden, t_weights_2)
    t_mse_target = torch.tensor(mse_target.value, requires_grad=False, dtype=torch.float)
    t_xent_target = torch.tensor(xent_target.value, requires_grad=False, dtype=torch.int64)

    t_loss_1 = F.mse_loss(t_predicted, t_mse_target, reduction="sum")
    t_loss_2 = F.cross_entropy(t_predicted, t_xent_target, reduction="sum")
    t_loss = t_loss_1 + t_loss_2

    np.testing.assert_almost_equal(loss.value, t_loss.detach().numpy(), decimal=4)

    expected_matches = {
        weights_1: t_weights_1,
        weights_2: t_weights_2,
        hidden: t_hidden,
        predicted: t_predicted,
        loss_1: t_loss_1,
        loss_2: t_loss_2,
        loss: t_loss,
    }

    for expected_tensor in expected_matches.values():
        expected_tensor.retain_grad()

    loss.backward()
    t_loss.backward()

    for actual_tensor, expected_tensor in expected_matches.items():
        np.testing.assert_almost_equal(actual_tensor.grad, expected_tensor.grad, decimal=4)


def test_three(num_iterations: int = 10_000, learning_rate: float = 1e-3):
    weights = tensor(np.random.randn(2, 2), requires_grad=True)
    target_fn = lambda x: np.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], axis=1)

    for iteration in range(num_iterations):
        x = tensor(np.random.randn(5, 2), requires_grad=False)
        target = tensor(target_fn(x.value), requires_grad=False)
        predicted = matmul(x, weights)

        loss = mse_loss(predicted, target)
        loss.backward()

        weights.value -= learning_rate * weights.grad
        weights.grad = np.zeros_like(weights.value)

    np.testing.assert_almost_equal(weights.value, np.array([[1, 1], [1, -1]]), decimal=4)


if __name__ == "__main__":
    test_one()
    test_two()
    test_three()
    print("All tests passed")
