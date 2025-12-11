#!/usr/bin/env python3
"""
Preprocessing script to extract video clips and save as numpy arrays.

This script eliminates the CPU bottleneck in training by pre-extracting all
annotated clips from videos. Clips are saved as uint8 numpy arrays with
spatial transformations applied (resize + center crop), requiring only
normalization at load time.

Usage:
    python preprocess_clips.py

Expected output:
    preprocessed_clips/
        train/
            {label}/
                {clip_id}.npy
        val/
            {label}/
                {clip_id}.npy
        test/
            {label}/
                {clip_id}.npy
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import av
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_video_frames_selective(path: str, start_frame: int, end_frame: int) -> List[np.ndarray]:
    """
    Read specific frames from a video file using PyAV.

    Args:
        path: Path to the video file
        start_frame: First frame to read (inclusive)
        end_frame: Last frame to read (inclusive)

    Returns:
        List of frames as numpy arrays (H, W, 3) in RGB format
    """
    frames = []

    try:
        container = av.open(path)

        for packet_idx, frame in enumerate(container.decode(video=0)):
            if packet_idx > end_frame:
                break
            if packet_idx >= start_frame:
                frames.append(frame.to_ndarray(format="rgb24"))

        container.close()
    except Exception as e:
        raise RuntimeError(f"Error reading video {path}: {e}")

    return frames


def uniform_sample_indices(total_frames: int, num_samples: int) -> List[int]:
    """
    Generate uniformly spaced indices for temporal sampling.

    Args:
        total_frames: Total number of frames available
        num_samples: Number of frames to sample

    Returns:
        List of frame indices
    """
    if total_frames <= num_samples:
        return list(range(total_frames))

    step = total_frames / num_samples
    indices = [int(i * step) for i in range(num_samples)]
    return indices


def resize_and_center_crop(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Resize and center crop a frame to the target size.

    This matches the VideoMAE preprocessing:
    - Resize shortest edge to size
    - Center crop to (size, size)

    Args:
        frame: Input frame as numpy array (H, W, 3)
        size: Target size for both dimensions

    Returns:
        Preprocessed frame as numpy array (size, size, 3) in uint8
    """
    # Convert to PIL for consistent preprocessing
    pil_image = Image.fromarray(frame)

    # Resize: shortest edge to target size
    w, h = pil_image.size
    if h < w:
        new_h = size
        new_w = int(w * (size / h))
    else:
        new_w = size
        new_h = int(h * (size / w))

    pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to (size, size)
    w, h = pil_image.size
    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size

    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert back to numpy (uint8)
    return np.array(pil_image, dtype=np.uint8)


def preprocess_single_clip(
    video_path: str,
    start_frame: int,
    end_frame: int,
    num_frames: int = 16,
    size: int = 224
) -> np.ndarray:
    """
    Extract and preprocess a single clip from a video.

    Args:
        video_path: Path to the source video
        start_frame: First frame of the clip
        end_frame: Last frame of the clip
        num_frames: Number of frames to sample uniformly
        size: Target spatial size (width and height)

    Returns:
        Preprocessed clip as numpy array (num_frames, size, size, 3) in uint8
    """
    # Read the clip frames
    frames = read_video_frames_selective(video_path, start_frame, end_frame)

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path} [{start_frame}:{end_frame}]")

    # Uniform temporal sampling
    indices = uniform_sample_indices(len(frames), num_frames)
    sampled_frames = [frames[i] for i in indices]

    # Apply spatial transformations
    preprocessed_frames = [resize_and_center_crop(frame, size) for frame in sampled_frames]

    # Stack into array (T, H, W, C)
    clip_array = np.stack(preprocessed_frames, axis=0)

    return clip_array


def load_annotations(data_dir: str, split: str) -> List[dict]:
    """
    Load annotations for a specific split.

    Args:
        data_dir: Root directory containing the dataset
        split: One of 'train', 'val', 'test'

    Returns:
        List of annotation dictionaries
    """
    annotation_file = os.path.join(data_dir, f"{split}.json")

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    return annotations


def preprocess_split(
    data_dir: str,
    split: str,
    output_dir: str,
    num_frames: int = 16,
    size: int = 224
) -> Tuple[int, int]:
    """
    Preprocess all clips for a specific split.

    Args:
        data_dir: Root directory containing the dataset
        split: One of 'train', 'val', 'test'
        output_dir: Directory to save preprocessed clips
        num_frames: Number of frames per clip
        size: Target spatial size

    Returns:
        Tuple of (successful_count, failed_count)
    """
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}")

    # Load annotations
    annotations = load_annotations(data_dir, split)
    print(f"Found {len(annotations)} clips in {split} split")

    # Create output directory
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    successful = 0
    failed = 0

    # Process each clip
    for idx, ann in enumerate(tqdm(annotations, desc=f"Preprocessing {split}")):
        try:
            # Extract clip information
            video_path = os.path.join(data_dir, ann['video'])
            start_frame = ann['start_frame']
            end_frame = ann['end_frame']
            label = ann['label']

            # Create label directory
            label_dir = os.path.join(split_output_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)

            # Generate clip filename (use index for uniqueness)
            clip_filename = f"clip_{idx:05d}.npy"
            output_path = os.path.join(label_dir, clip_filename)

            # Skip if already exists
            if os.path.exists(output_path):
                successful += 1
                continue

            # Preprocess the clip
            clip_array = preprocess_single_clip(
                video_path,
                start_frame,
                end_frame,
                num_frames=num_frames,
                size=size
            )

            # Validate shape
            expected_shape = (num_frames, size, size, 3)
            if clip_array.shape != expected_shape:
                raise ValueError(f"Unexpected shape {clip_array.shape}, expected {expected_shape}")

            # Save to disk
            np.save(output_path, clip_array)
            successful += 1

        except Exception as e:
            print(f"\nError processing clip {idx} from {ann.get('video', 'unknown')}: {e}")
            failed += 1
            continue

    print(f"\n{split} split complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    return successful, failed


def main():
    """Main preprocessing function."""
    # Configuration
    DATA_DIR = "Olympic Boxing Punch Classification Video Dataset"  # Adjust if needed
    OUTPUT_DIR = "preprocessed_clips"
    NUM_FRAMES = 16
    SIZE = 224

    print("="*60)
    print("Video Clip Preprocessing")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Frames per clip: {NUM_FRAMES}")
    print(f"Spatial size: {SIZE}x{SIZE}")
    print(f"Output format: uint8 numpy arrays")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split
    total_successful = 0
    total_failed = 0

    for split in ['train', 'val', 'test']:
        successful, failed = preprocess_split(
            DATA_DIR,
            split,
            OUTPUT_DIR,
            num_frames=NUM_FRAMES,
            size=SIZE
        )
        total_successful += successful
        total_failed += failed

    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Total clips processed: {total_successful + total_failed}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")

    # Calculate storage size
    if total_successful > 0:
        sample_file = None
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file.endswith('.npy'):
                    sample_file = os.path.join(root, file)
                    break
            if sample_file:
                break

        if sample_file:
            file_size_mb = os.path.getsize(sample_file) / (1024 * 1024)
            total_size_gb = (file_size_mb * total_successful) / 1024
            print(f"\nStorage usage:")
            print(f"  Per clip: {file_size_mb:.2f} MB")
            print(f"  Total: {total_size_gb:.2f} GB")

    print(f"\nPreprocessed clips saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
