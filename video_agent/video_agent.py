
import os
import time
from pathlib import Path

import cv2

from video_agent.src.branches.inference import (
    load_model,
    analyze_video as run_inference
)


SUPPORTED_FORMATS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

MIN_DURATION_SECONDS = 1.0
MIN_FRAME_COUNT = 16


def classify_prediction(ai_probability, disagreement):
    """Classify the video as AI_GENERATED, AUTHENTIC or UNCERTAIN."""

    if disagreement > 0.4:
        return {
            "verdict": "UNCERTAIN",
            "confidence": "LOW",
            "reason": "High disagreement between detection branches."
        }

    if 0.35 <= ai_probability <= 0.65:
        return {
            "verdict": "UNCERTAIN",
            "confidence": "LOW",
            "reason": "AI probability is in the uncertain range."
        }

    if ai_probability > 0.65:
        return {
            "verdict": "AI_GENERATED",
            "confidence": "HIGH" if ai_probability > 0.85 else "MEDIUM",
            "reason": "Visual analysis indicates AI-generated content."
        }

    return {
        "verdict": "AUTHENTIC",
        "confidence": "HIGH" if ai_probability < 0.15 else "MEDIUM",
        "reason": "Visual analysis indicates authentic camera footage."
    }


def validate_video(video_path):
    """Validate video file, metadata and readability."""

    if not video_path.exists():
        return False, "Video file does not exist.", None

    if not video_path.is_file():
        return False, "Provided path is not a file.", None

    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, "Unsupported video format.", None

    if video_path.stat().st_size == 0:
        return False, "Video file is empty.", None

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return False, "Video cannot be opened or is corrupted.", None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return False, "Invalid video metadata.", None

    duration = total_frames / fps

    if total_frames < MIN_FRAME_COUNT:
        cap.release()
        return False, "Video contains fewer than 16 frames.", None

    if duration < MIN_DURATION_SECONDS:
        cap.release()
        return False, "Video is too short for reliable analysis.", None

    readable = False

    for frame_index in [0, total_frames // 2, total_frames - 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if success and frame is not None:
            readable = True
            break

    cap.release()

    if not readable:
        return False, "Video frames cannot be decoded.", None

    metadata = {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2)
    }

    return True, None, metadata


def analyze_video(video_path):
    """Main interface for the TrueLensAI Video Agent."""

    video_path = Path(video_path)

    # 1. Validate video
    valid, error, metadata = validate_video(video_path)

    if not valid:
        return {
            "status": "error",
            "agent": "video_agent",
            "error": error
        }

    # 2. Validate model checkpoint
    checkpoint = os.getenv("AEGIS_CHECKPOINT")

    if not checkpoint or not Path(checkpoint).is_file():
        return {
            "status": "error",
            "agent": "video_agent",
            "error": "Model checkpoint not found."
        }

    try:
        device = "cpu"

        # 3. Load model
        model = load_model(checkpoint, device)

        # 4. Run AEGIS inference
        start_time = time.time()

        results, _ = run_inference(
            str(video_path),
            model,
            device
        )

        processing_time = round(time.time() - start_time, 2)

        # 5. Classify prediction
        prediction_result = classify_prediction(
            results["ai_probability"],
            results["disagreement"]
        )

        # 6. Return structured response
        return {
            "status": "success",
            "agent": "video_agent",
            "video_path": str(video_path),

            "verdict": prediction_result["verdict"],
            "confidence": prediction_result["confidence"],
            "reason": prediction_result["reason"],

            "prediction": results["ai_probability"],
            "pixel_probability": results["pixel_prob"],
            "motion_probability": results["motion_prob"],
            "branch_disagreement": results["disagreement"],
            "smoothness_score": results["smoothness"],
            "frame_scores": results["frame_scores"],

            "video_metadata": metadata,
            "processing_time_seconds": processing_time
        }

    except Exception as error:
        return {
            "status": "error",
            "agent": "video_agent",
            "error": str(error)
        }
