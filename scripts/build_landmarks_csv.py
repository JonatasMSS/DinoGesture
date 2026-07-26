"""Gera um CSV de landmarks normalizados a partir das sessões coletadas."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


LANDMARK_COUNT = 468
NOSE_TIP_INDEX = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_ROOT = PROJECT_ROOT / "data" / "recordings"
OUTPUT_CSV = PROJECT_ROOT / "data" / "collected_landmarks.csv"
REJECTED_CSV = PROJECT_ROOT / "data" / "collected_landmarks_rejected.csv"
HEADER = ["label", "session_id", "frame_index"] + [
    f"{axis}_{index}"
    for index in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
]


def flatten_landmarks(face_landmarks: Any) -> list[float]:
    """Centraliza os pontos pela ponta do nariz para reduzir deslocamento na tela."""
    landmarks = face_landmarks.landmark
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError("invalid_landmark_count")
    anchor = landmarks[NOSE_TIP_INDEX]
    anchor_coordinates = (float(anchor.x or 0.0), float(anchor.y or 0.0), float(anchor.z or 0.0))
    return [
        0.0 if value is None else float(value) - anchor_coordinate
        for landmark in landmarks
        for value, anchor_coordinate in zip(
            (landmark.x, landmark.y, landmark.z), anchor_coordinates, strict=True
        )
    ]


def frame_index(path: Path, fallback: int) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return fallback


def process_frame(image_path: Path, detector: Any) -> tuple[list[float] | None, str | None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "unreadable_image"

    results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = results.multi_face_landmarks or []
    if not faces:
        return None, "no_face"
    if len(faces) != 1:
        return None, "multiple_faces"
    try:
        return flatten_landmarks(faces[0]), None
    except ValueError as error:
        return None, str(error)


def session_directories(recordings_root: Path):
    for label_directory in sorted(recordings_root.iterdir()) if recordings_root.exists() else []:
        if not label_directory.is_dir() or label_directory.name.startswith("."):
            continue
        for session_directory in sorted(label_directory.iterdir()):
            if session_directory.is_dir():
                yield label_directory.name, session_directory


def _build_dataset(
    recordings_root: Path,
    output_csv: Path,
    rejected_csv: Path,
    detector: Any,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    labels = Counter()
    sessions = images = valid = rejected = 0

    with (
        output_csv.open("w", newline="", encoding="utf-8") as output_file,
        rejected_csv.open("w", newline="", encoding="utf-8") as rejected_file,
    ):
        writer = csv.writer(output_file)
        rejected_writer = csv.writer(rejected_file)
        writer.writerow(HEADER)
        rejected_writer.writerow(["label", "session_id", "frame_index", "path", "reason"])

        session_paths = list(session_directories(recordings_root))
        total_images = sum(
            len(list(session_directory.glob("*.png")))
            for _, session_directory in session_paths
        )
        with tqdm(total=total_images, desc="Gerando CSV", unit="frame") as progress:
            for label, session_directory in session_paths:
                sessions += 1
                for position, image_path in enumerate(sorted(session_directory.glob("*.png")), start=1):
                    images += 1
                    coordinates, reason = process_frame(image_path, detector)
                    index = frame_index(image_path, position)
                    if reason:
                        rejected += 1
                        rejected_writer.writerow([label, session_directory.name, index, image_path, reason])
                    else:
                        writer.writerow([label, session_directory.name, index, *coordinates])
                        labels[label] += 1
                        valid += 1
                    progress.update(1)

    return {
        "sessions": sessions,
        "images": images,
        "valid": valid,
        "rejected": rejected,
        "labels": dict(sorted(labels.items())),
    }


def build_dataset(
    recordings_root: Path = RECORDINGS_ROOT,
    output_csv: Path = OUTPUT_CSV,
    rejected_csv: Path = REJECTED_CSV,
    detector: Any | None = None,
) -> dict[str, Any]:
    if detector is not None:
        return _build_dataset(recordings_root, output_csv, rejected_csv, detector)

    import mediapipe as mp

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=False,
    ) as face_mesh:
        return _build_dataset(recordings_root, output_csv, rejected_csv, face_mesh)


def main() -> None:
    summary = build_dataset()
    print(f"Sessões: {summary['sessions']}")
    print(f"Frames válidos: {summary['valid']}")
    print(f"Frames descartados: {summary['rejected']}")
    print(f"Distribuição: {summary['labels']}")


if __name__ == "__main__":
    main()
