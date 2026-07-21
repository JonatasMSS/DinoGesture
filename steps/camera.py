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
