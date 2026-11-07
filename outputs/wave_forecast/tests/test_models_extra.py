"""Shape and ordering checks for the GRU and CNN-BiGRU baselines and the threshold-weighted CRPS loss."""
import unittest

import torch

from wavecast.models import GRUNet, CNNBiGRU, Transformer, TiDE, crps_loss, twcrps_loss, crps_loss_masked, add_te_head, te_quantiles

CFG = {"hidden": 32, "layers": 2, "heads": 4, "degree": 8, "dropout": 0.1}


class ExtraModelTests(unittest.TestCase):
    def check(self, cls):
        torch.manual_seed(0)
        model = cls(10, 56, 112, CFG)
        past, future, y = torch.randn(3, 56, 10), torch.randn(3, 112, 10), torch.rand(3, 112) * 3
        q = model(past, future)
        self.assertEqual(tuple(q.shape), (3, 112, 99))
        self.assertTrue(bool((q.diff(dim=-1) >= -1e-6).all()), "quantiles must be ordered")
        self.assertTrue(bool((q >= 0).all()), "wave height quantiles must be nonnegative")
        loss = crps_loss(q, y, model.head.tau) + twcrps_loss(q, y, model.head.tau, 2.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_gru(self):
        self.check(GRUNet)

    def test_cnn_bigru(self):
        self.check(CNNBiGRU)

    def test_twcrps_zero_below_threshold(self):
        tau = (torch.arange(99) + .5) / 99
        q = torch.full((1, 99), 1.0)
        y = torch.tensor([0.5])
        self.assertAlmostEqual(float(twcrps_loss(q, y, tau, 2.5)), 0.0)

    def test_te_head_all_models(self):
        """Period head: same shape as the Hs head, positive and ordered, gradients reach both heads."""
        for cls in (Transformer, TiDE, GRUNet, CNNBiGRU):
            model = add_te_head(cls(10, 56, 112, CFG), CFG)
            past, future = torch.randn(2, 56, 10), torch.randn(2, 112, 10)
            q = model(past, future)
            qt = te_quantiles(model)
            self.assertEqual(tuple(qt.shape), tuple(q.shape))
            self.assertTrue(bool((qt >= 0).all()) and bool((qt[..., 1:] >= qt[..., :-1] - 1e-6).all()))
            y = torch.rand(2, 112) * 3
            yt = torch.full((2, 112), float("nan")); yt[0, :50] = 7.0
            loss = crps_loss(q, y, model.head.tau) + crps_loss_masked(qt / 5, yt / 5, model.head.tau)
            loss.backward()
            self.assertIsNotNone(model.te_head.linear.weight.grad)
            self.assertIsNotNone(model.head.linear.weight.grad)

    def test_masked_crps_ignores_missing(self):
        tau = (torch.arange(99) + .5) / 99
        q = torch.rand(1, 4, 99).sort(-1).values
        y = torch.tensor([[1.0, float("nan"), 0.3, float("nan")]])
        full = crps_loss(q[:, [0, 2]], y[:, [0, 2]], tau)
        self.assertAlmostEqual(float(crps_loss_masked(q, y, tau)), float(full), places=6)
        self.assertEqual(float(crps_loss_masked(q, torch.full((1, 4), float("nan")), tau)), 0.0)


if __name__ == "__main__":
    unittest.main()
