import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# Minimum PPE required — "protectuve boots" is a typo in the original dataset labels
REQUIRED_PPE = {
    "protective helmet",
    "jacket",
    "glove",
    "protectuve boots",
}


def parse_args():
    parser = argparse.ArgumentParser(description="PPE Detection using YOLOv8")
    parser.add_argument("--source", type=str, default="0",
                        help="Video file, image, directory, or 0 for webcam")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Path to model weights (.pt file)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--save", action="store_true",
                        help="Save output to disk as output.mp4")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="Disable real-time display window")
    parser.set_defaults(show=True)
    return parser.parse_args()


def draw_status(frame, missing):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
    if missing:
        label = f"MISSING: {', '.join(m.title() for m in sorted(missing))}"
        color = (0, 60, 255)
    else:
        label = "All required PPE detected"
        color = (0, 200, 60)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return frame


def run_stream(model, source, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    print("Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=args.conf, persist=True, verbose=False)

        detected = {results[0].names[int(b.cls)].lower() for b in results[0].boxes}
        missing = REQUIRED_PPE - detected

        annotated = results[0].plot()
        annotated = draw_status(annotated, missing)

        if args.show:
            cv2.imshow("PPE Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                )
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
        print("Saved to output.mp4")
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    model = YOLO(str(model_path))

    try:
        source = int(args.source)
        run_stream(model, source, args)
    except ValueError:
        src = Path(args.source)
        if src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            run_stream(model, str(src), args)
        else:
            model.predict(
                source=str(src), conf=args.conf, show=args.show, save=args.save
            )


if __name__ == "__main__":
    main()
