import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "PyTorch is required to test the federated aggregators."
    ) from exc

from utils.federated_aggregators import (
    FedAdagradAggregator,
    FedAvgAggregator,
    FedAvgMAggregator,
    FedYogiAggregator,
    create_server_aggregator,
)


class FederatedAggregatorTests(unittest.TestCase):
    def test_fedavg_returns_averaged_state_without_aliasing(self):
        current = {"weight": torch.tensor([0.0])}
        averaged = {"weight": torch.tensor([2.0])}

        result = FedAvgAggregator().aggregate(current, averaged)

        self.assertTrue(torch.equal(result["weight"], averaged["weight"]))
        result["weight"].add_(1.0)
        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0])))

    def test_fedavgm_keeps_momentum_across_rounds(self):
        aggregator = FedAvgMAggregator(server_lr=1.0, momentum=0.9)

        first = aggregator.aggregate(
            {"weight": torch.tensor([0.0])},
            {"weight": torch.tensor([1.0])},
        )
        second = aggregator.aggregate(
            first,
            {"weight": torch.tensor([2.0])},
        )

        self.assertTrue(torch.allclose(first["weight"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(second["weight"], torch.tensor([2.9])))

    def test_fedadagrad_accumulates_squared_updates(self):
        aggregator = FedAdagradAggregator(server_lr=0.1, tau=1e-6)

        first = aggregator.aggregate(
            {"weight": torch.tensor([0.0])},
            {"weight": torch.tensor([1.0])},
        )
        second = aggregator.aggregate(
            first,
            {"weight": first["weight"] + 1.0},
        )

        expected_first = torch.tensor([0.1 / 1.000001])
        expected_second = expected_first + (
            0.1 / (torch.sqrt(torch.tensor(2.0)) + 1e-6)
        )
        self.assertTrue(torch.allclose(first["weight"], expected_first))
        self.assertTrue(torch.allclose(second["weight"], expected_second))

    def test_fedyogi_applies_adaptive_update(self):
        aggregator = FedYogiAggregator(
            server_lr=0.01,
            beta1=0.0,
            beta2=0.0,
            tau=0.1,
        )

        result = aggregator.aggregate(
            {"weight": torch.tensor([0.0])},
            {"weight": torch.tensor([2.0])},
        )

        self.assertTrue(
            torch.allclose(result["weight"], torch.tensor([0.02 / 2.1]))
        )

    def test_non_floating_state_uses_averaged_value(self):
        aggregator = FedAvgMAggregator()

        result = aggregator.aggregate(
            {"num_batches_tracked": torch.tensor(2, dtype=torch.long)},
            {"num_batches_tracked": torch.tensor(5, dtype=torch.long)},
        )

        self.assertEqual(result["num_batches_tracked"].item(), 5)
        self.assertEqual(result["num_batches_tracked"].dtype, torch.long)

    def test_factory_is_case_insensitive(self):
        self.assertIsInstance(
            create_server_aggregator("FedYogi"),
            FedYogiAggregator,
        )

    def test_factory_builds_every_supported_algorithm(self):
        expected_types = {
            "fedavg": FedAvgAggregator,
            "fedavgm": FedAvgMAggregator,
            "fedyogi": FedYogiAggregator,
            "fedadagrad": FedAdagradAggregator,
        }

        for name, expected_type in expected_types.items():
            with self.subTest(name=name):
                self.assertIsInstance(
                    create_server_aggregator(name),
                    expected_type,
                )

    def test_factory_rejects_unknown_algorithm(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported federated algorithm 'unknown'",
        ):
            create_server_aggregator("unknown")

    def test_shape_mismatch_has_clear_error(self):
        aggregator = FedAdagradAggregator()

        with self.assertRaisesRegex(ValueError, "current shape"):
            aggregator.aggregate(
                {"weight": torch.zeros(2)},
                {"weight": torch.zeros(3)},
            )


if __name__ == "__main__":
    unittest.main()
