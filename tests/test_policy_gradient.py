import torch
from policy_gradient import compute_returns, policy_improvement
import unittest


class TestPolicyGradient(unittest.TestCase):
    def test_compute_returns(self):
        self.assertEqual(
            compute_returns([1, 1, 1, 1, 1]),
            [4.90099501, 3.9403989999999998, 2.9701, 1.99, 1.0],
        )

    def test_policy_improvement(self):
        log_prob = torch.tensor(-2.0)
        log_prob.requires_grad = True
        self.assertAlmostEqual(
            policy_improvement([log_prob, log_prob / 2.0], [1, 1]), 0.707, 3
        )


if __name__ == "__main__":
    unittest.main()
