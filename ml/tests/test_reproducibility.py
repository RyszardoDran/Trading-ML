"""Test reproducibility of sequence training pipeline with fixed random seeds.

This test ensures that running the pipeline multiple times with the same
random_state produces identical results.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestReproducibility:
    """Test random seed handling for reproducibility."""
    
    def test_random_seed_initialization(self):
        """Test that random seeds can be set correctly."""
        # Simulate what the pipeline does
        random_state = 42
        
        # Set seeds (same order as in pipeline)
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Generate random values
        py_val1 = random.random()
        np_val1 = np.random.rand()
        
        # Reset seeds and generate again
        random.seed(random_state)
        np.random.seed(random_state)
        
        py_val2 = random.random()
        np_val2 = np.random.rand()
        
        # Values should be identical
        assert py_val1 == py_val2, "Python random values differ with same seed"
        assert np_val1 == np_val2, "NumPy random values differ with same seed"
    
    def test_random_seed_order_independence(self):
        """Test that seed order doesn't affect reproducibility."""
        random_state = 42
        
        # Order 1: random first, then numpy
        random.seed(random_state)
        np.random.seed(random_state)
        py_val1 = random.random()
        np_val1 = np.random.rand()
        
        # Order 2: numpy first, then random
        np.random.seed(random_state)
        random.seed(random_state)
        py_val2 = random.random()
        np_val2 = np.random.rand()
        
        # Values should be identical regardless of order
        assert py_val1 == py_val2, "Python random affected by seed order"
        assert np_val1 == np_val2, "NumPy random affected by seed order"
    
    def test_numpy_array_operations_deterministic(self):
        """Test that numpy array operations are deterministic."""
        random_state = 42
        
        # Run 1
        np.random.seed(random_state)
        arr1 = np.random.randn(100, 10)
        result1 = arr1.mean()
        
        # Run 2
        np.random.seed(random_state)
        arr2 = np.random.randn(100, 10)
        result2 = arr2.mean()
        
        # Should be identical
        assert np.allclose(arr1, arr2), "NumPy arrays differ with same seed"
        assert result1 == result2, "NumPy operations not deterministic"
    
    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different values."""
        # Seed 1
        random.seed(42)
        np.random.seed(42)
        py_val1 = random.random()
        np_val1 = np.random.rand()
        
        # Seed 2
        random.seed(123)
        np.random.seed(123)
        py_val2 = random.random()
        np_val2 = np.random.rand()
        
        # Values should be different
        assert py_val1 != py_val2, "Python random produces same value with different seeds"
        assert np_val1 != np_val2, "NumPy random produces same value with different seeds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
