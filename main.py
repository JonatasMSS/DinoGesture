import argparse

import cv2
import mediapipe as mp

from pipeline import FrameContext, Pipeline
from steps import (
    CaptureFrameStep,
    DetectFaceStep,
    DisplayFrameStep,
    DrawLogicPointsStep,
    DrawLandmarksStep,
    LogicalCommandStep,
    MirrorFrameStep,
)


def positive_threshold(value: str) -> float:
    threshold = float(value)
    if threshold <= 0:
        raise argparse.ArgumentTypeError("O limiar deve ser maior que zero.")
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exibe os 468 landmarks faciais da webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice da camera usada pelo OpenCV (padrao: 0).",
    )
    parser.add_argument(
        "--show-logic-points",
        action="store_true",
        help="Destaca os pontos usados pelo agente lógico.",
    )
    parser.add_argument(
        "--mouth-threshold",
        type=positive_threshold,
        default=0.08,
        help="Aumento mínimo normalizado para boca aberta.",
    )
    parser.add_argument(
        "--brow-threshold",
        type=positive_threshold,
        default=0.04,
        help="Aumento mínimo normalizado para sobrancelhas levantadas.",
    )
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera {args.camera}.")

    face_mesh = mp.solutions.face_mesh
    try:
        with face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
        ) as detector:
            steps = [
                CaptureFrameStep(camera),
                MirrorFrameStep(),
                DetectFaceStep(detector),
                LogicalCommandStep(args.mouth_threshold, args.brow_threshold),
                DrawLandmarksStep(),
            ]
            if args.show_logic_points:
                steps.append(DrawLogicPointsStep())
            # PredictFaceCommandStep(model) permanece desativado neste teste lógico.
            steps.append(DisplayFrameStep())
            pipeline = Pipeline(steps)
            while True:
                context = FrameContext()
                pipeline.run(context)
                if context.should_exit:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
