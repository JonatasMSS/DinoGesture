from typing import Any

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
        context.landmarks = (
            results.multi_face_landmarks[0].landmark
            if results.multi_face_landmarks
            else None
        )
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
        cv2.imshow(self.title, context.frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            context.should_exit = True
            return None
        return context
