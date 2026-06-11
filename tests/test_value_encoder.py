import unittest

import torch

from model.value_encoder import ValueEncoder


class ValueEncoderPathFilmTests(unittest.TestCase):
    def test_zero_initialized_path_film_preserves_numeric_encoding(self):
        torch.manual_seed(123)
        base = ValueEncoder(
            d_model=8,
            frozen_lm_dim=4,
            n_fourier_feats=4,
            fourier_learnable=False,
        )
        torch.manual_seed(123)
        film = ValueEncoder(
            d_model=8,
            frozen_lm_dim=4,
            n_fourier_feats=4,
            fourier_learnable=False,
            numeric_path_film=True,
        )
        film.mantissa_encoder.load_state_dict(base.mantissa_encoder.state_dict())
        film.exponent_embed.load_state_dict(base.exponent_embed.state_dict())

        node_types = ["number", "number"]
        raw_values = [0.5, 12.0]
        path_context = torch.randn(2, 8)

        base_out = base(node_types, raw_values)
        film_out = film(node_types, raw_values, path_context=path_context)

        self.assertTrue(torch.allclose(base_out, film_out, atol=1e-6))

    def test_zero_initialized_path_gamma_preserves_numeric_encoding(self):
        torch.manual_seed(123)
        base = ValueEncoder(
            d_model=8,
            frozen_lm_dim=4,
            n_fourier_feats=4,
            fourier_learnable=False,
        )
        torch.manual_seed(123)
        gamma = ValueEncoder(
            d_model=8,
            frozen_lm_dim=4,
            n_fourier_feats=4,
            fourier_learnable=False,
            numeric_path_gamma=True,
        )
        gamma.mantissa_encoder.load_state_dict(base.mantissa_encoder.state_dict())
        gamma.exponent_embed.load_state_dict(base.exponent_embed.state_dict())

        node_types = ["number", "number"]
        raw_values = [0.5, 12.0]
        path_context = torch.randn(2, 8)

        base_out = base(node_types, raw_values)
        gamma_out = gamma(node_types, raw_values, path_context=path_context)

        self.assertTrue(torch.allclose(base_out, gamma_out, atol=1e-6))

    def test_path_film_requires_path_context_for_numbers(self):
        encoder = ValueEncoder(
            d_model=8,
            frozen_lm_dim=4,
            n_fourier_feats=4,
            fourier_learnable=False,
            numeric_path_film=True,
        )

        with self.assertRaisesRegex(ValueError, "path_context"):
            encoder(["number"], [1.0])

    def test_path_gamma_requires_path_context_for_numbers(self):
        encoder = ValueEncoder(
            d_model=8,
            frozen_lm_dim=4,
            n_fourier_feats=4,
            fourier_learnable=False,
            numeric_path_gamma=True,
        )

        with self.assertRaisesRegex(ValueError, "path_context"):
            encoder(["number"], [1.0])

    def test_path_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ValueEncoder(
                d_model=8,
                frozen_lm_dim=4,
                n_fourier_feats=4,
                fourier_learnable=False,
                numeric_path_film=True,
                numeric_path_gamma=True,
            )


if __name__ == "__main__":
    unittest.main()
