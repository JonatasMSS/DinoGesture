import argparse
from pathlib import Path

import cv2

from pipeline import FrameContext, Pipeline
from steps import (
    CaptureFrameStep,
    DetectFaceStep,
    DinoActionStep,
    DisplayFrameStep,
    DrawLandmarksStep,
    LogicalCommandStep,
    MirrorFrameStep,
)
from utils.chrome import launch_dino


def positive_threshold(value: str) -> float:
    threshold = float(value)
    if threshold <= 0:
        raise argparse.ArgumentTypeError("O limiar deve ser maior que zero.")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exibe os 468 landmarks faciais da webcam."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--show-logic-points", action="store_true")
    parser.add_argument("--mouth-threshold", type=positive_threshold, default=0.08)
    parser.add_argument("--brow-threshold", type=positive_threshold, default=0.04)
    parser.add_argument("--game-control", action="store_true")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--chrome-path", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.debug_mode and not args.game_control:
        raise ValueError("--debug-mode requer --game-control.")


def build_steps(args: argparse.Namespace, camera, detector) -> list:
    steps = [
        CaptureFrameStep(camera),
        MirrorFrameStep(),
        DetectFaceStep(detector),
        LogicalCommandStep(args.mouth_threshold, args.brow_threshold),
    ]
    if args.game_control:
        steps.append(DinoActionStep(debug_mode=args.debug_mode))
        if args.debug_mode:
            steps.extend([DrawLandmarksStep(show_logic_points=True), DisplayFrameStep()])
    else:
        steps.extend(
            [
                DrawLandmarksStep(show_logic_points=args.show_logic_points),
                DisplayFrameStep(),
            ]
        )
    return steps


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    if args.game_control:
        launch_dino(args.chrome_path)
        print("Chrome Dino aberto. Calibrando a camera; encerre com Ctrl+C.")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera {args.camera}.")

    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh
    try:
        with face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
        ) as detector:
            pipeline = Pipeline(build_steps(args, camera, detector))
            while True:
                context = FrameContext()
                pipeline.run(context)
                if context.should_exit:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
