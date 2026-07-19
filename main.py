import argparse

import cv2
import mediapipe as mp

from pipeline import FrameContext, Pipeline
from steps import (
    CaptureFrameStep,
    DetectFaceStep,
    DinoGameControllerStep,
    DisplayFrameStep,
    DrawLandmarksStep,
    DrawLogicPointsStep,
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
    parser.add_argument(
        "--game-control",
        action="store_true",
        help="Ativa o controle do jogo Chrome Dino.",
    )
    args = parser.parse_args()

    camera = None
    detector = None
    game_controller = None

    try:
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a camera {args.camera}.")

        if args.game_control:
            from scripts.script_dinogame import open_dino_game

            game_controller = open_dino_game()

        face_mesh = mp.solutions.face_mesh
        detector = face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
        )

        steps = [
            CaptureFrameStep(camera),
            MirrorFrameStep(),
            DetectFaceStep(detector),
            LogicalCommandStep(args.mouth_threshold, args.brow_threshold),
        ]
        if game_controller is not None:
            steps.append(DinoGameControllerStep(game_controller))

        steps.append(DrawLandmarksStep())
        if args.show_logic_points:
            steps.append(DrawLogicPointsStep())

        steps.append(DisplayFrameStep())
        pipeline = Pipeline(steps)

        while True:
            context = FrameContext()
            pipeline.run(context)
            if context.should_exit:
                break
    finally:
        if game_controller is not None:
            game_controller.close()
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
