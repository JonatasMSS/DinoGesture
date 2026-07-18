from steps import PredictFaceCommandStep
from model.Model import Model
import argparse

import cv2
import mediapipe as mp

from pipeline import FrameContext, Pipeline
from steps import (
    CaptureFrameStep,
    DetectFaceStep,
    DisplayFrameStep,
    DrawLandmarksStep,
    MirrorFrameStep,
)


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
    args = parser.parse_args()
    model = Model(r"C:\Users\gears\OneDrive\Desktop\DinoGesture\notebooks\Trainings\RandomForest.pkl")

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
            pipeline = Pipeline(
                [
                    CaptureFrameStep(camera),
                    MirrorFrameStep(),
                    DetectFaceStep(detector),
                    DrawLandmarksStep(),
                    PredictFaceCommandStep(model),
                    DisplayFrameStep(),
                ]
            )
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
