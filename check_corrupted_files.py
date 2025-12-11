#!/usr/bin/env python3
"""
Parallel corruption checker for preprocessed .npy files.
Uses one worker per label directory for maximum speed.
"""

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def check_directory(label_dir: Path) -> Tuple[str, List[str], List[str], int]:
    """
    Check all .npy files in a label directory for corruption.

    Args:
        label_dir: Path to label directory containing .npy files

    Returns:
        Tuple of (label_name, empty_files, eoferror_files, valid_count)
    """
    label_name = label_dir.name
    empty_files = []
    eoferror_files = []
    other_errors = []
    valid_count = 0

    # Get all .npy files in this directory
    npy_files = list(label_dir.glob("*.npy"))

    # Progress bar for this label
    for npy_file in tqdm(npy_files, desc=f"Checking {label_name}", position=None, leave=True):
        try:
            # Check if file is empty (0 bytes)
            if os.path.getsize(npy_file) == 0:
                empty_files.append(str(npy_file))
                continue

            # Try to load the file
            data = np.load(npy_file)

            # Check if data is empty
            if data.size == 0:
                empty_files.append(str(npy_file))
            else:
                valid_count += 1

        except EOFError:
            eoferror_files.append(str(npy_file))
        except Exception as e:
            other_errors.append(f"{npy_file}: {str(e)}")

    # Combine all error types for return
    all_errors = empty_files + eoferror_files + other_errors

    return label_name, empty_files, eoferror_files, valid_count


def main():
    """Main function to check all preprocessed clips."""
    # Configuration
    PREPROCESSED_DIR = Path("preprocessed_clips_3")

    if not PREPROCESSED_DIR.exists():
        print(f"Error: Directory {PREPROCESSED_DIR} does not exist!")
        return

    print("="*70)
    print("PARALLEL CORRUPTION CHECKER")
    print("="*70)
    print(f"Scanning directory: {PREPROCESSED_DIR}")
    print()

    # Find all label directories
    label_dirs = [d for d in PREPROCESSED_DIR.iterdir() if d.is_dir()]

    if not label_dirs:
        print(f"No label directories found in {PREPROCESSED_DIR}")
        return

    print(f"Found {len(label_dirs)} label directories")
    print(f"Labels: {sorted([d.name for d in label_dirs])}")
    print()

    # Use one worker per label directory (or max CPU count)
    num_workers = min(len(label_dirs), cpu_count())
    print(f"Using {num_workers} parallel workers")
    print("="*70)
    print()

    # Process directories in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(check_directory, label_dirs)

    # Aggregate results
    total_valid = 0
    total_empty = 0
    total_eoferror = 0
    all_empty_files = []
    all_eoferror_files = []

    print("\n" + "="*70)
    print("RESULTS BY LABEL:")
    print("="*70)

    for label_name, empty_files, eoferror_files, valid_count in sorted(results):
        total_valid += valid_count
        total_empty += len(empty_files)
        total_eoferror += len(eoferror_files)
        all_empty_files.extend(empty_files)
        all_eoferror_files.extend(eoferror_files)

        error_count = len(empty_files) + len(eoferror_files)
        total_count = valid_count + error_count

        status = "✓" if error_count == 0 else "✗"
        print(f"{status} {label_name:6s}: {valid_count:5d} valid, {error_count:3d} corrupted (total: {total_count})")

    print("="*70)
    print()

    # Summary
    total_files = total_valid + total_empty + total_eoferror
    print("SUMMARY:")
    print("="*70)
    print(f"Total files:        {total_files:6d}")
    print(f"Valid files:        {total_valid:6d} ({100*total_valid/total_files:.2f}%)")
    print(f"Empty files:        {total_empty:6d} ({100*total_empty/total_files:.2f}%)")
    print(f"EOFError files:     {total_eoferror:6d} ({100*total_eoferror/total_files:.2f}%)")
    print(f"Total corrupted:    {total_empty + total_eoferror:6d} ({100*(total_empty + total_eoferror)/total_files:.2f}%)")
    print("="*70)
    print()

    # List all corrupted files
    if all_empty_files:
        print(f"\nEMPTY FILES ({len(all_empty_files)} total):")
        print("-"*70)
        for f in sorted(all_empty_files):
            print(f"  {f}")
        print()

    if all_eoferror_files:
        print(f"\nEOFERROR FILES ({len(all_eoferror_files)} total):")
        print("-"*70)
        for f in sorted(all_eoferror_files):
            print(f"  {f}")
        print()

    # Generate removal command
    if all_empty_files or all_eoferror_files:
        print("\nTO REMOVE CORRUPTED FILES, RUN:")
        print("-"*70)
        all_corrupted = all_empty_files + all_eoferror_files
        for f in sorted(all_corrupted):
            print(f'rm "{f}"')
        print()
        print("OR run this single command:")
        print(f"find {PREPROCESSED_DIR} -name '*.npy' -size 0 -delete")
        print("="*70)


if __name__ == "__main__":
    main()
