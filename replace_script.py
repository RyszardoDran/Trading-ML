import re

file_path = 'ml/src/pipelines/sequence_training_pipeline.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new function code
new_function = r'''def create_sequences(
    features: pd.DataFrame,
    targets: pd.Series,
    window_size: int = 100,
    session: str = "all",
    custom_start: int = None,
    custom_end: int = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create sliding windows of features with corresponding targets.

    Optimized for memory:
    1. Drops NaNs from inputs first
    2. Filters by session BEFORE creating full matrix
    3. Uses stride tricks to avoid copies until necessary

    Args:
        features: Per-candle feature matrix (n_samples, n_features)
        targets: Binary labels aligned with features
        window_size: Number of candles in each window
        session: Session filter ('london', 'ny', 'asian', 'london_ny', 'all', 'custom')
        custom_start: Start hour for custom session
        custom_end: End hour for custom session

    Returns:
        X: (n_windows, window_size * n_features) array
        y: (n_windows,) array of binary labels
        timestamps: DatetimeIndex of window end times
    """
    # 1. Pre-clean data to avoid checking NaNs on huge X matrix later
    if features.isnull().values.any():
        logger.warning("Input features contain NaNs. Dropping rows with NaNs...")
        features = features.dropna()
    
    # Align features and targets
    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    if len(features) < window_size:
        raise ValueError(f"Need at least {window_size} samples, got {len(features)}")

    n_features = features.shape[1]
    n_windows = len(features) - window_size + 1

    # Prepare arrays
    features_array = np.ascontiguousarray(features.values, dtype=np.float32)
    targets_array = targets.values.astype(np.int32)
    timestamps_array = features.index.values

    # 2. Calculate timestamps and targets for all potential windows
    # Timestamps aligned to the END of the window
    timestamp_indices = np.arange(window_size - 1, window_size - 1 + n_windows)
    timestamps = pd.DatetimeIndex(timestamps_array[timestamp_indices])
    
    # Targets aligned to the END of the window
    y = targets_array[window_size - 1 : window_size - 1 + n_windows]

    # 3. Apply Session Filter (Indices)
    # We calculate the mask on timestamps BEFORE creating the heavy X matrix
    hours = timestamps.hour
    if session == "london":
        mask = (hours >= 8) & (hours < 16)
    elif session == "ny":
        mask = (hours >= 13) & (hours < 22)
    elif session == "asian":
        mask = (hours >= 0) & (hours < 9)
    elif session == "london_ny":
        mask = (hours >= 8) & (hours < 22)
    elif session == "all":
        mask = np.ones(len(timestamps), dtype=bool)
    elif session == "custom":
        if custom_start is None or custom_end is None:
            raise ValueError("Must provide custom_start and custom_end for 'custom' session")
        if custom_start < custom_end:
            mask = (hours >= custom_start) & (hours < custom_end)
        else:
            mask = (hours >= custom_start) | (hours < custom_end)
    else:
        raise ValueError(f"Unknown session: {session}")

    if mask.sum() == 0:
        logger.warning(f"Session filter '{session}' removed all data!")
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    logger.info(f"Session filter '{session}': keeping {mask.sum():,} / {len(timestamps):,} windows ({mask.mean():.1%})")

    # Filter y and timestamps
    y = y[mask]
    timestamps = timestamps[mask]
    
    # 4. Create X only for valid windows
    # Use stride tricks to get a view, then slice with mask, then reshape
    from numpy.lib.stride_tricks import as_strided

    shape = (n_windows, window_size, n_features)
    strides = (features_array.strides[0], features_array.strides[0], features_array.strides[1])

    try:
        # Create view of ALL windows (no copy)
        windowed_view = as_strided(features_array, shape=shape, strides=strides, writeable=False)
        
        # Apply mask to view (creates copy of ONLY valid windows)
        # Shape becomes (n_valid, window_size, n_features)
        X_valid = windowed_view[mask]
        
        # Flatten (creates copy of valid windows)
        # Shape becomes (n_valid, window_size * n_features)
        X = X_valid.reshape(X_valid.shape[0], -1)
        
    except Exception as e:
        logger.warning(f"Stride trick optimization failed: {e}. Falling back to loop...")
        # Fallback: iterate only over valid indices
        valid_indices = np.where(mask)[0]
        n_valid = len(valid_indices)
        X = np.zeros((n_valid, window_size * n_features), dtype=np.float32)
        
        for i, idx in enumerate(valid_indices):
            # idx is the window index. Window starts at idx in features_array
            X[i] = features_array[idx : idx + window_size].flatten()

    return X, y, timestamps'''

# Regex to find the old function
# It starts with def create_sequences and ends before def make_target
pattern = re.compile(r'def create_sequences\(.*?\n\) -> Tuple\[np\.ndarray, np\.ndarray, pd\.DatetimeIndex\]:.*?return X, y, timestamps', re.DOTALL)

# Check if we can find the pattern
match = pattern.search(content)
if match:
    print("Found function to replace.")
    new_content = content.replace(match.group(0), new_function)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully replaced function.")
else:
    print("Could not find function to replace.")
    # Debug: print a snippet where it should be
    start_idx = content.find("def create_sequences")
    if start_idx != -1:
        print("Snippet around start:")
        print(content[start_idx:start_idx+200])
