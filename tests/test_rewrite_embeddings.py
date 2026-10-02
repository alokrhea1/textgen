"""Model-free contracts. Intentionally not executed during implementation."""
import unittest
import os
from unittest.mock import Mock

import numpy as np

from modules.sentence_rewrite.embeddings import (
    EmbeddingConfig, LateInteractionEncoder, maxsim, symmetric_maxsim, score_many,
)


class ScoringTests(unittest.TestCase):
    def test_score_many_matches_pair_scores(self):
        query = np.array([[2., 0.], [1., 1.]])
        documents = [np.array([[-3., 0.]]), np.eye(2), np.array([[1., -1.], [-1., 1.], [1., 0.]])]
        for mode, function in [("directional", maxsim), ("symmetric", symmetric_maxsim)]:
            np.testing.assert_allclose(score_many(query, documents, mode),
                                       [function(query, document) for document in documents], atol=1e-6)
        self.assertEqual(score_many(query, []), [])
        with self.assertRaises(ValueError):
            score_many(query, documents, "pooled")

    @unittest.skipUnless(os.environ.get("REWRITE_TEST_CUDA") == "1", "Explicit CUDA scoring test opt-in")
    def test_cuda_ragged_padding_and_oversized_tiling(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(47)
        query = rng.normal(size=(2051, 7)).astype(np.float32)
        documents = [rng.normal(size=(length, 7)).astype(np.float32) for length in [1, 31, 2049]]
        # Final pair exceeds 4M similarity elements and uses tiled GPU scoring.
        for mode in ["directional", "symmetric"]:
            np.testing.assert_allclose(score_many(query, documents, mode, "cuda"),
                                       score_many(query, documents, mode, "cpu"), atol=2e-5, rtol=2e-5)
        # Padded zeros must not defeat genuine negative cosine matches.
        negative = [np.array([[-1., 0.]]), np.array([[-1., 0.], [-1., 0.]])]
        np.testing.assert_allclose(score_many(np.array([[1., 0.]]), negative, "symmetric", "cuda"), [-1., -1.])

    def test_multiple_blocks_match_dense_exact_scoring(self):
        # Cross both tile boundaries with distinct query and document lengths.
        rng = np.random.default_rng(41)
        query = rng.normal(size=(529, 17)).astype(np.float32)
        document = rng.normal(size=(1043, 17)).astype(np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        document /= np.linalg.norm(document, axis=1, keepdims=True)
        dense = np.clip(query @ document.T, -1, 1)
        expected_forward = dense.max(axis=1).mean()
        expected_symmetric = (expected_forward + dense.max(axis=0).mean()) / 2
        self.assertAlmostEqual(maxsim(query, document), float(expected_forward), places=6)
        self.assertAlmostEqual(symmetric_maxsim(query, document), float(expected_symmetric), places=6)

    def test_forward_coverage_differs_from_reverse_coverage(self):
        short = np.array([[1., 0.]])
        long = np.eye(2)
        self.assertAlmostEqual(maxsim(short, long), 1.)
        self.assertAlmostEqual(maxsim(long, short), .5)
        self.assertAlmostEqual(symmetric_maxsim(short, long), .75)
        self.assertEqual(symmetric_maxsim(short, long), symmetric_maxsim(long, short))

    def test_cosine_normalizes_and_preserves_negative_scores(self):
        self.assertAlmostEqual(maxsim(np.array([[3., 0.]]), np.array([[-2., 0.]])), -1.)

    def test_invalid_matrices_are_rejected(self):
        for invalid in (np.zeros((0, 2)), np.zeros((1, 2)), np.array([[np.nan, 1.]])):
            with self.assertRaises(ValueError):
                maxsim(invalid, np.eye(2))
        with self.assertRaises(ValueError):
            maxsim(np.eye(2), np.eye(3))


class BackendContractTests(unittest.TestCase):
    def encoder(self):
        encoder = LateInteractionEncoder(EmbeddingConfig(max_tokens=8))
        model = Mock()
        model.prompts = {"query": "Q ", "document": "D "}
        model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": list(range(len(text.split()) + 2))}
        model.encode_document.return_value = [np.array([[3., 0.], [0., 4.]])]
        model.encode_query.return_value = [np.array([[0., 2.]])]
        encoder._model = model
        return encoder, model

    def test_roles_output_dtype_and_normalization(self):
        encoder, model = self.encoder()
        result = encoder.encode_sentence("one sentence")
        np.testing.assert_allclose(result, np.eye(2))
        self.assertEqual(result.dtype, np.float32)
        encoder.encode_query("one sentence")
        model.encode_query.assert_called_once()
        model.encode_document.assert_called_once()
        self.assertTrue(model.encode_document.call_args.kwargs["normalize_embeddings"])

    def test_overlong_input_is_rejected_before_encoding(self):
        encoder, model = self.encoder()
        with self.assertRaises(ValueError):
            encoder.encode_query("one two three four five six seven eight")
        model.encode_query.assert_not_called()

    def test_empty_collection_and_padding_free_matrices(self):
        encoder, model = self.encoder()
        self.assertEqual(encoder.encode_documents([]), [])
        model.encode_document.assert_not_called()
        self.assertEqual(encoder.token_count("one two"), 5)


if __name__ == "__main__":
    unittest.main()
