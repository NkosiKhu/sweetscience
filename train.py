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
import math

from sklearn.metrics import classification_report


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
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    train_paths = []
    val_paths = []
    test_paths = []
    for label in os.listdir("preprocessed_clips/train/"):
        paths = (lambda x: [f"preprocessed_clips/train/{x}/{p}" 
                        for p in os.listdir(f"preprocessed_clips/train/{x}")])(label)
        paths_count = len(paths)
        train_ind = math.floor(paths_count * 0.8)
        val_ind = train_ind + math.floor(paths_count * 0.1)
        test_ind = val_ind + math.floor(paths_count * 0.1)
        train_paths.extend(paths[:train_ind])
        val_paths.extend(paths[train_ind:val_ind])
        test_paths.extend(paths[val_ind:])
        
    def __init__(self, split: str):
        self.split = split
        
        
    def __len__(self):
        if self.split == "train":
            return len(self.train_paths)
        elif self.split == "val":
            return len(self.val_paths)
        elif self.split == "test":
            return len(self.test_paths)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, idx):
        if self.split == "train":
            path = self.train_paths[idx]
        elif self.split == "val":
            path = self.val_paths[idx]
        elif self.split == "test":
            path = self.test_paths[idx]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        clip = np.load(path)
        
        # convert to float and scale to 0-1
        clip = clip.astype(np.float32) / 255.0
        
        # image net mean/std
        clip = (clip - self.mean) / self.std
        
        #reorder to (T,C,H,W)
        clip = clip.transpose(0,3,1,2)
        
        #convert to tensor
        clip = torch.from_numpy(clip)
        
        return {
            "pixel_values": clip,
            "labels": torch.tensor(LABEL2ID[path.split("/")[-2]], dtype=torch.long) 
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

    # Overall metrics
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    # Per-class metrics
    report = classification_report(
        labels, preds,
        target_names=list(LABEL2ID.keys()),
        output_dict=True,
        zero_division=0
    )

    # Add per-class F1 to metrics
    per_class_metrics = {}
    for label_name in LABEL2ID.keys():
        per_class_metrics[f"f1_{label_name}"] = report[label_name]["f1-score"]
        per_class_metrics[f"precision_{label_name}"] = report[label_name]["precision"]
        per_class_metrics[f"recall_{label_name}"] = report[label_name]["recall"]

    return {
        "accuracy": acc,
        "macro_f1": f1,
        **per_class_metrics
    }


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
