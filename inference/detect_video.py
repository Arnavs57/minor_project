"""
Video inference for marine waste detection.

Runs YOLOv8 on a video file (or webcam) and saves an annotated video.
Optionally applies underwater preprocessing per frame.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")


def detect_video(
    source: str,
    weights: str,
    output_dir: str = "results",
    conf: float = 0.5,
    iou: float = 0.45,
    preprocess: bool = False,
    save: bool = True,
    show_labels: bool = True,
) -> list:
    """
    Run marine waste detection on a video file or webcam stream.

    Uses YOLOv8's native video support: pass the video path as source and
    set save=True to write the annotated video to output_dir.

    Args:
        source: Path to video file (.mp4, .avi, etc.) or "0" for default webcam.
        weights: Path to model weights (e.g. runs/detect/train/weights/best.pt).
        output_dir: Directory where the annotated video is saved (when save=True).
        conf: Confidence threshold (0–1).
        iou: IoU threshold for NMS.
        preprocess: If True, apply underwater preprocessing to each frame (slower).
        save: If True, save the annotated video.
        show_labels: If True, draw class and confidence on boxes.

    Returns:
        List of Ultralytics Results (one per frame when not streaming; stream=True returns iterator).
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    YOLO = _get_yolo()
    model = YOLO(str(weights_path))

    # Webcam: source is "0" or device index
    if source.isdigit() or source.strip().lower() in ("0", "1", "2"):
        source = int(source) if source.isdigit() else source

    if preprocess:
        # Frame-by-frame: read with OpenCV, preprocess, then run YOLO (no native video path for that)
        import cv2
        from preprocessing.pipeline import preprocess_image
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = Path(output_dir) / "video"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "annotated.mp4"
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
        frame_idx = 0
        all_results = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                enhanced = preprocess_image(frame, color_correct=True, clahe=True, dehaze=True)
                res = model.predict(enhanced, conf=conf, iou=iou, verbose=False)
                all_results.extend(res)
                if save and len(res) and res[0].plot is not None:
                    plot_img = res[0].plot()
                    writer.write(plot_img)
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        print(f"Saved annotated video: {out_file} ({frame_idx} frames)")
        return all_results

    # No preprocessing: YOLO handles video natively
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=save,
        project=output_dir,
        name="video",
        exist_ok=True,
        show_labels=show_labels,
    )
    return list(results)


def main():
    parser = argparse.ArgumentParser(
        description="Detect marine waste in a video file or webcam (YOLOv8)",
    )
    parser.add_argument("--source", "-s", required=True, help='Video path (e.g. video.mp4) or "0" for webcam')
    parser.add_argument("--weights", "-w", required=True, help="Path to model weights (e.g. best.pt)")
    parser.add_argument("--output", "-o", default="results", help="Output directory for annotated video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0–1)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--preprocess", action="store_true", help="Apply underwater preprocessing per frame (slower)")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")
    args = parser.parse_args()

    try:
        from utils.helpers import load_config
        cfg = load_config()
        inf = cfg.get("inference", {})
        conf = getattr(args, "conf", None) or inf.get("confidence_threshold", 0.5)
        iou = getattr(args, "iou", None) or inf.get("iou_threshold", 0.45)
    except Exception:
        conf, iou = args.conf, args.iou

    detect_video(
        source=args.source,
        weights=args.weights,
        output_dir=args.output,
        conf=conf,
        iou=iou,
        preprocess=args.preprocess,
        save=not args.no_save,
    )
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
