
from scripts.build_landmarks_csv import flatten_landmarks
from typing import Any
from model.Model import Model

import cv2

from pipeline import FrameContext


class CaptureFrameStep:
    def __init__(self, camera: Any) -> None:
        self.camera = camera

    def process(self, context: FrameContext) -> FrameContext:
        success, context.frame = self.camera.read()
        if not success:
            raise RuntimeError("Nao foi possivel ler um frame da camera.")
        return context




class MirrorFrameStep:
    def process(self, context: FrameContext) -> FrameContext:
        context.frame = cv2.flip(context.frame, 1)
        return context


class DetectFaceStep:
    def __init__(self, detector: Any) -> None:
        self.detector = detector

    def process(self, context: FrameContext) -> FrameContext:
        results = self.detector.process(
            cv2.cvtColor(context.frame, cv2.COLOR_BGR2RGB)
        )
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            context.landmarks = face_landmarks.landmark
            context.features = flatten_landmarks(face_landmarks)
        else:
            context.landmarks = None
            context.features = None
        return context

class PredictFaceCommandStep:
    def __init__(self,model:Model):
        self.predictor = model

    def process(self,context:FrameContext) -> FrameContext:
        if context.features is not None:
            context.prediction = self.predictor.predict([context.features])[0]
        return context


class DrawLandmarksStep:
    def process(self, context: FrameContext) -> FrameContext:
        if context.landmarks:
            height, width = context.frame.shape[:2]
            for landmark in context.landmarks:
                cv2.circle(
                    context.frame,
                    (int(landmark.x * width), int(landmark.y * height)),
                    1,
                    (0, 255, 0),
                    -1,
                )
        return context


class DisplayFrameStep:
    def __init__(self, title: str = "Landmarks faciais") -> None:
        self.title = title

    def process(self, context: FrameContext) -> FrameContext | None:
        if context.prediction is not None:
            text = f"Predição: {context.prediction}"
            color = (0, 255, 0)
        else:
            text = "Sem face detectada"
            color = (0, 0, 255)
        cv2.putText(context.frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow(self.title, context.frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            context.should_exit = True
            return None
        return context
