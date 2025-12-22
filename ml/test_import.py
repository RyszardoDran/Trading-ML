#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Try import
try:
    from src.filters.regime_filter import filter_predictions_by_regime, should_trade, get_adaptive_threshold
    print("✅ SUCCESS: All filter functions imported")
    print("\nFilter functions available:")
    print(f"  - filter_predictions_by_regime: {callable(filter_predictions_by_regime)}")
    print(f"  - should_trade: {callable(should_trade)}")
    print(f"  - get_adaptive_threshold: {callable(get_adaptive_threshold)}")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
