"""
Unit tests for the count_tokens utility.
"""

import os
import tempfile
import numpy as np
import pytest

from LM_training.tokenizer.cli.count_tokens import (
    count_tokens_in_npy,
    calculate_training_metrics,
    format_number,
)


class TestCountTokens:
    """Tests for counting tokens in .npy files."""

    def test_count_tokens_in_npy(self):
        """Test counting tokens in a temporary .npy file."""
        # Create a temporary .npy file with known token count
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            test_tokens = np.array([1, 2, 3, 4, 5, 10, 20, 30], dtype=np.uint16)
            np.save(tmp_file.name, test_tokens)
            tmp_path = tmp_file.name

        try:
            num_tokens = count_tokens_in_npy(tmp_path)
            assert num_tokens == 8, f"Expected 8 tokens, got {num_tokens}"
        finally:
            os.unlink(tmp_path)

    def test_count_tokens_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            count_tokens_in_npy("/nonexistent/path/file.npy")

    def test_count_tokens_large_file(self):
        """Test counting tokens in a larger file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            # Create a larger dataset (1 million tokens)
            test_tokens = np.random.randint(0, 10000, size=1_000_000, dtype=np.uint16)
            np.save(tmp_file.name, test_tokens)
            tmp_path = tmp_file.name

        try:
            num_tokens = count_tokens_in_npy(tmp_path)
            assert num_tokens == 1_000_000, f"Expected 1,000,000 tokens, got {num_tokens}"
        finally:
            os.unlink(tmp_path)


class TestCalculateTrainingMetrics:
    """Tests for training metrics calculation."""

    def test_basic_metrics(self):
        """Test basic training metrics calculation without max_iters."""
        metrics = calculate_training_metrics(
            num_tokens=100_000,
            batch_size=32,
            context_length=256,
            max_iters=None
        )

        assert metrics["total_tokens"] == 100_000
        assert metrics["tokens_per_batch"] == 32 * 256  # 8,192
        assert metrics["max_possible_steps"] == 100_000 // 8_192  # 12
        assert "epochs" not in metrics

    def test_metrics_with_max_iters(self):
        """Test training metrics with max_iters for epoch calculation."""
        metrics = calculate_training_metrics(
            num_tokens=100_000,
            batch_size=32,
            context_length=256,
            max_iters=100
        )

        tokens_per_batch = 32 * 256  # 8,192
        total_tokens_seen = tokens_per_batch * 100  # 819,200
        expected_epochs = total_tokens_seen / 100_000  # 8.192

        assert metrics["max_iters"] == 100
        assert metrics["total_tokens_seen"] == total_tokens_seen
        assert abs(metrics["epochs"] - expected_epochs) < 0.001

    def test_metrics_less_than_one_epoch(self):
        """Test when training doesn't complete a full epoch."""
        metrics = calculate_training_metrics(
            num_tokens=1_000_000,
            batch_size=32,
            context_length=256,
            max_iters=100
        )

        expected_epochs = (32 * 256 * 100) / 1_000_000  # 0.8192
        assert metrics["epochs"] < 1.0
        assert abs(metrics["epochs"] - expected_epochs) < 0.001

    def test_metrics_exactly_one_epoch(self):
        """Test when training completes exactly one epoch."""
        num_tokens = 100_000
        batch_size = 32
        context_length = 256
        tokens_per_batch = batch_size * context_length
        max_iters = num_tokens // tokens_per_batch

        metrics = calculate_training_metrics(
            num_tokens=num_tokens,
            batch_size=batch_size,
            context_length=context_length,
            max_iters=max_iters
        )

        # Should be close to 1.0 (might be slightly less due to integer division)
        assert 0.9 < metrics["epochs"] <= 1.0

    def test_metrics_multiple_epochs(self):
        """Test when training goes through multiple epochs."""
        metrics = calculate_training_metrics(
            num_tokens=100_000,
            batch_size=32,
            context_length=256,
            max_iters=5000
        )

        expected_epochs = (32 * 256 * 5000) / 100_000  # 409.6
        assert metrics["epochs"] > 1.0
        assert abs(metrics["epochs"] - expected_epochs) < 0.1


class TestFormatNumber:
    """Tests for number formatting utility."""

    def test_format_integer(self):
        """Test formatting integers with commas."""
        assert format_number(1000) == "1,000"
        assert format_number(1000000) == "1,000,000"
        assert format_number(12345) == "12,345"

    def test_format_float(self):
        """Test formatting floats with two decimal places."""
        assert format_number(1234.5678) == "1,234.57"
        assert format_number(100.1) == "100.10"

    def test_format_small_numbers(self):
        """Test formatting small numbers."""
        assert format_number(5) == "5"
        assert format_number(42) == "42"
        assert format_number(3.14) == "3.14"
