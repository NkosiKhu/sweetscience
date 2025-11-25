import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import av  # pip install av
from transformers import (
    AutoImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)
import evaluate  # pip install evaluate


# -------------------------
# 1. Label mapping (FACTS boxing classes)
# -------------------------

# Polish labels from Olympic Boxing dataset to English codes
POLISH_TO_ENGLISH: Dict[str, str] = {
    "Głowa lewą ręką": "LHHP",      # Left Hand Head Punch
    "Głowa prawą ręką": "RHHP",     # Right Hand Head Punch
    "Chybienie lewą ręką": "LHMP",  # Left Hand Missed Punch
    "Chybienie prawą ręką": "RHMP", # Right Hand Missed Punch
    "Blok lewą ręką": "LHBlP",      # Left Hand Block Punch
    "Blok prawą ręką": "RHBlP",     # Right Hand Block Punch
    "Korpus lewą ręką": "LHBP",     # Left Hand Body Punch
    "Korpus prawą ręką": "RHBP",    # Right Hand Body Punch
}

LABEL2ID: Dict[str, int] = {
    "LHHP": 0,   # Left Hand Head Punch
    "RHHP": 1,   # Right Hand Head Punch
    "LHMP": 2,   # Left Hand Missed Punch
    "RHMP": 3,   # Right Hand Missed Punch
    "LHBlP": 4,  # Left Hand Block Punch
    "RHBlP": 5,  # Right Hand Block Punch
    "LHBP": 6,   # Left Hand Body Punch
    "RHBP": 7,   # Right Hand Body Punch
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NUM_FRAMES = 16  # VideoMAE default / FACTS config
IMAGE_SIZE = 224


# -------------------------
# 2. Video reading + frame sampling (uniform temporal subsampling)
# -------------------------

def read_video_pyav(path: str) -> np.ndarray:
    """
    Read *all* frames from a video file as RGB uint8 array of shape (T, H, W, 3).

    WARNING: This function is memory-inefficient and loads all frames.
    Use read_video_frames_selective() instead for clip-based loading.
    """
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)  # (T, H, W, 3)


def read_video_frames_selective(path: str, start_frame: int, end_frame: int) -> np.ndarray:
    """
    Read only a specific range of frames from a video file.
    Much more memory-efficient than reading the entire video.

    Args:
        path: Path to the video file
        start_frame: First frame index to read (inclusive)
        end_frame: Last frame index to read (inclusive)

    Returns:
        np.ndarray of shape (num_frames, H, W, 3) with RGB uint8 frames
    """
    container = av.open(path)
    video_stream = container.streams.video[0]

    frames = []

    # We need to decode from the beginning or from a keyframe
    # For simplicity and correctness, we decode all frames but only keep the ones we need
    # This is still much more memory-efficient than keeping the entire decoded video in memory
    for packet_idx, frame in enumerate(container.decode(video=0)):
        if packet_idx > end_frame:
            # We've passed the end, stop decoding
            break
        if packet_idx >= start_frame:
            # This frame is in our target range, keep it
            frames.append(frame.to_ndarray(format="rgb24"))

    container.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {path} in range [{start_frame}, {end_frame}]")

    return np.stack(frames)  # (num_frames, H, W, 3)


def uniform_sample_indices(num_total: int, num_target: int) -> np.ndarray:
    """
    Uniformly sample num_target indices from [0, num_total-1].
    If num_total < num_target, we pad by repeating the last frame.
    """
    if num_total >= num_target:
        indices = np.linspace(0, num_total - 1, num=num_target, dtype=np.int64)
    else:
        # upsample: linearly spaced indices then clip to last frame
        indices = np.linspace(0, num_total - 1, num=num_target, dtype=np.float32)
        indices = np.clip(indices, 0, num_total - 1).astype(np.int64)
    return indices


# -------------------------
# 3. BoxingDataset — reads from Olympic Boxing dataset structure
# -------------------------

class BoxingDataset(Dataset):
    """
    Loads video clips from the Olympic Boxing Punch Classification dataset.

    Dataset structure:
    - base_dir/
      - task_kam2_gh078416/
        - annotations.json  # Contains tracks with labels and frame ranges
        - data/
          - GH078416.mp4    # Full video file
      - task_kam2_gh088416/
        ...

    Each annotations.json contains tracks where each track represents a punch clip
    with a label (Polish) and frame range (bounding boxes across frames).

    Supports two modes:
    1. On-demand loading (use_preprocessed=False): Decodes video clips at runtime (slower, CPU-bound)
    2. Preprocessed loading (use_preprocessed=True): Loads pre-extracted numpy arrays (faster, GPU-bound)
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        image_processor: AutoImageProcessor,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        use_preprocessed: bool = False,
        preprocessed_dir: str = "preprocessed_clips",
    ):
        """
        Args:
            dataset_dir: Path to "Olympic Boxing Punch Classification Video Dataset" folder
            split: One of "train", "val", or "test"
            image_processor: HuggingFace image processor for VideoMAE
            train_ratio: Proportion of data for training (default 0.7)
            val_ratio: Proportion of data for validation (default 0.15)
            seed: Random seed for reproducible splits
            use_preprocessed: If True, load from preprocessed numpy arrays instead of decoding videos
            preprocessed_dir: Directory containing preprocessed clips (default "preprocessed_clips")
        """
        self.items: List[Dict[str, Any]] = []
        self.image_processor = image_processor
        self.use_preprocessed = use_preprocessed
        self.preprocessed_dir = preprocessed_dir

        # Parse all task folders
        all_samples = []
        task_folders = sorted([d for d in os.listdir(dataset_dir) if d.startswith('task_')])

        for task_folder in task_folders:
            task_path = os.path.join(dataset_dir, task_folder)
            annotations_file = os.path.join(task_path, "annotations.json")
            data_folder = os.path.join(task_path, "data")

            if not os.path.exists(annotations_file):
                continue

            # Find the video file in data folder
            video_files = [f for f in os.listdir(data_folder) if f.endswith('.mp4')]
            if len(video_files) == 0:
                continue
            video_path = os.path.join(data_folder, video_files[0])

            # Parse annotations
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract tracks (each track is a labeled punch clip)
            if len(data) > 0 and 'tracks' in data[0]:
                tracks = data[0]['tracks']

                for track in tracks:
                    label_polish = track.get('label', '')

                    # Convert Polish label to English code
                    if label_polish not in POLISH_TO_ENGLISH:
                        print(f"Warning: Unknown label '{label_polish}' in {task_folder}, skipping")
                        continue

                    label_english = POLISH_TO_ENGLISH[label_polish]
                    label_id = LABEL2ID[label_english]

                    # Extract frame range from shapes
                    shapes = track.get('shapes', [])
                    if not shapes:
                        continue

                    frames = [s['frame'] for s in shapes if 'frame' in s and not s.get('outside', False)]
                    if not frames:
                        continue

                    start_frame = min(frames)
                    end_frame = max(frames)

                    all_samples.append({
                        'video_path': video_path,
                        'label_id': label_id,
                        'label_name': label_english,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'task_folder': task_folder,
                    })

        # Split data into train/val/test
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))

        n_train = int(len(all_samples) * train_ratio)
        n_val = int(len(all_samples) * val_ratio)

        if split == "train":
            selected_indices = indices[:n_train]
        elif split == "val":
            selected_indices = indices[n_train:n_train + n_val]
        elif split == "test":
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")

        # Add preprocessed clip paths to each sample
        selected_samples = [all_samples[i] for i in selected_indices]
        for idx, sample in enumerate(selected_samples):
            # Preprocessed clips are organized as: {split}/{label}/{clip_id}.npy
            label_id = sample['label_id']
            clip_filename = f"clip_{idx:05d}.npy"
            preprocessed_path = os.path.join(preprocessed_dir, split, str(label_id), clip_filename)
            sample['preprocessed_path'] = preprocessed_path

        self.items = selected_samples

        if len(self.items) == 0:
            raise RuntimeError(f"No samples found for split={split}")

        print(f"Loaded {len(self.items)} samples for {split} split")

        # Print label distribution
        label_counts = {}
        for item in self.items:
            label = item['label_name']
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution for {split}: {label_counts}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.items[idx]
        label_id = entry['label_id']

        if self.use_preprocessed:
            # FAST PATH: Load preprocessed numpy array
            preprocessed_path = entry['preprocessed_path']

            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(
                    f"Preprocessed clip not found: {preprocessed_path}\n"
                    f"Please run preprocess_clips.py first to generate preprocessed clips."
                )

            # Load preprocessed clip (NUM_FRAMES, H, W, 3) in uint8
            clip_frames = np.load(preprocessed_path)  # (16, 224, 224, 3)

            # Apply only normalization (resize + crop already done during preprocessing)
            # The image_processor expects a list of PIL Images or numpy arrays
            processed = self.image_processor(
                list(clip_frames),  # sequence of frames
                return_tensors="pt",
                do_resize=False,  # Already resized during preprocessing
                do_center_crop=False,  # Already cropped during preprocessing
            )
            pixel_values = processed["pixel_values"].squeeze(0)  # drop batch dim

        else:
            # SLOW PATH: On-demand video decoding (original behavior)
            video_path = entry['video_path']
            start_frame = entry['start_frame']
            end_frame = entry['end_frame']

            # 1) Decode only the required frame range (MEMORY EFFICIENT!)
            clip_frames = read_video_frames_selective(
                video_path, start_frame, end_frame
            )  # (num_frames, H, W, 3), uint8

            # 2) Uniform temporal subsampling to NUM_FRAMES
            T = clip_frames.shape[0]
            indices = uniform_sample_indices(T, NUM_FRAMES)
            clip_frames = clip_frames[indices]  # (NUM_FRAMES, H, W, 3)

            # 3) HF VideoMAE preprocessing: resize, center-crop, rescale, ImageNet norm
            processed = self.image_processor(
                list(clip_frames),  # sequence of frames
                return_tensors="pt",
            )
            pixel_values = processed["pixel_values"].squeeze(0)  # drop batch dim

        return {
            "pixel_values": pixel_values,
            "labels": label_id,
        }


# -------------------------
# 4. Data collator for video classification
# -------------------------

@dataclass
class VideoDataCollator:
    """
    Simple collator that stacks pixel_values and labels.
    Assumes all clips have same num_frames, H, W after preprocessing.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


# -------------------------
# 5. Metrics (accuracy + macro F1)
# -------------------------

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "macro_f1": f1}


# -------------------------
# 6. Main: load model, datasets, and train
# -------------------------

def main():
    # Point this at the Olympic Boxing dataset directory
    DATASET_DIR = "Olympic Boxing Punch Classification Video Dataset"

    # Pretrained VideoMAE base (self-supervised on K400)
    model_name = "MCG-NJU/videomae-base"

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # FACTS used batch_size=4, grad_accum=2, warmup_ratio=0.1, epochs=10
    # Learning rate is not rendered in the HTML; start with 1e-4 and tune around it.
    training_args = TrainingArguments(
        output_dir="./facts-boxing-videomae",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # effective batch size 8
        warmup_ratio=0.1,
        learning_rate=1e-4,
        weight_decay=0.05,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",  # or "wandb"/"tensorboard"
    )

    train_dataset = BoxingDataset(
        dataset_dir=DATASET_DIR,
        split="train",
        image_processor=image_processor,
    )
    val_dataset = BoxingDataset(
        dataset_dir=DATASET_DIR,
        split="val",
        image_processor=image_processor,
    )
    test_dataset = BoxingDataset(
        dataset_dir=DATASET_DIR,
        split="test",
        image_processor=image_processor,
    )

    data_collator = VideoDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate on test split
    test_metrics = trainer.evaluate(test_dataset)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
