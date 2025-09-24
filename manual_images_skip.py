import argparse
from pathlib import Path
import shutil
from detectors.yolov11_face_detector import YOLOv11FaceDetector


def ensure_dir(output_dir: Path):
    """Ensure that the output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)



def process_images(detector: YOLOv11FaceDetector, img_folder: Path, output_dir: Path):
    images = list(img_folder.glob("*.jpg"))

    if not images:
        print(f"No images found in: {img_folder}")
        return
    
    ensure_dir(output_dir)

    for image in images:
        detections = detector.detect_faces(str(image))
        if detections:
            detector.visualize(
                str(image),
                detections,
                save_path=str(output_dir / f"vis_{image.name}")
            )
            print(f"{image}: {detections}")
            
        else:
            print(f"Could not detect a face in: {image}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces in images using YOLOv11 and save results."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to folder containing input images"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="detected",
        help="Path to folder for saving detected images"
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=0.3,
        help="Confidence threshold for YOLO detector (default: 0.3)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="models/yolov11n-face.pt",
        help="Path to YOLOv11 face detection model weights"
    )

    args = parser.parse_args()

    img_folder = Path(args.input)
    output_dir = Path(args.output)

    detector = YOLOv11FaceDetector(model_path=args.model, conf_thresh=args.conf)
    process_images(detector, img_folder, output_dir)


if __name__ == "__main__":
    main()
