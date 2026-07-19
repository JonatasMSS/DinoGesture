
import math
from statistics import median
from typing import Any

from scripts.build_landmarks_csv import flatten_landmarks
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


MOUTH_POINTS = (13, 14)
EYE_CORNERS = (33, 263)
LEFT_BROW_POINTS = (70, 63, 105, 66, 107)
RIGHT_BROW_POINTS = (336, 296, 334, 293, 300)
LEFT_EYE_POINTS = (159, 145)
RIGHT_EYE_POINTS = (386, 374)


class LogicalCommandStep:
    def __init__(
        self,
        mouth_threshold: float = 0.08,
        brow_threshold: float = 0.04,
        calibration_frames: int = 24,
        confirmation_frames: int = 3,
    ) -> None:
        self.mouth_threshold = mouth_threshold
        self.brow_threshold = brow_threshold
        self.calibration_frames = calibration_frames
        self.confirmation_frames = confirmation_frames
        self.calibration_samples: list[dict[str, float]] = []
        self.baseline: dict[str, float] | None = None
        self.command = 0
        self.candidate: int | None = None
        self.candidate_frames = 0

    @staticmethod
    def _mean_y(landmarks: Any, indices: tuple[int, ...]) -> float:
        return sum(float(landmarks[index].y) for index in indices) / len(indices)

    def _measure(self, landmarks: Any) -> dict[str, float] | None:
        left_corner, right_corner = (landmarks[index] for index in EYE_CORNERS)
        eye_distance = math.hypot(
            float(right_corner.x) - float(left_corner.x),
            float(right_corner.y) - float(left_corner.y),
        )
        if eye_distance == 0:
            return None

        upper_lip, lower_lip = (landmarks[index] for index in MOUTH_POINTS)
        mouth_opening = math.hypot(
            float(lower_lip.x) - float(upper_lip.x),
            float(lower_lip.y) - float(upper_lip.y),
        ) / eye_distance
        left_brow_gap = (
            self._mean_y(landmarks, LEFT_EYE_POINTS)
            - self._mean_y(landmarks, LEFT_BROW_POINTS)
        ) / eye_distance
        right_brow_gap = (
            self._mean_y(landmarks, RIGHT_EYE_POINTS)
            - self._mean_y(landmarks, RIGHT_BROW_POINTS)
        ) / eye_distance
        return {
            "mouth_opening": mouth_opening,
            "left_brow_gap": left_brow_gap,
            "right_brow_gap": right_brow_gap,
        }

    def _finish_calibration(self) -> None:
        self.baseline = {
            name: median(sample[name] for sample in self.calibration_samples)
            for name in self.calibration_samples[0]
        }

    def _raw_command(self, measurements: dict[str, float]) -> int:
        assert self.baseline is not None
        mouth_delta = measurements["mouth_opening"] - self.baseline["mouth_opening"]
        left_brow_delta = measurements["left_brow_gap"] - self.baseline["left_brow_gap"]
        right_brow_delta = measurements["right_brow_gap"] - self.baseline["right_brow_gap"]
        measurements.update(
            mouth_delta=mouth_delta,
            left_brow_delta=left_brow_delta,
            right_brow_delta=right_brow_delta,
        )
        if mouth_delta >= self.mouth_threshold:
            return 1
        if left_brow_delta >= self.brow_threshold and right_brow_delta >= self.brow_threshold:
            return 2
        return 0

    def process(self, context: FrameContext) -> FrameContext:
        context.logic_measurements = None
        context.calibrating = self.baseline is None
        context.command = None
        if not context.landmarks:
            self.candidate = None
            self.candidate_frames = 0
            return context

        measurements = self._measure(context.landmarks)
        if measurements is None:
            return context
        context.logic_measurements = measurements

        if self.baseline is None:
            self.calibration_samples.append(measurements)
            if len(self.calibration_samples) == self.calibration_frames:
                self._finish_calibration()
                context.calibrating = False
                context.command = self.command
            return context

        raw_command = self._raw_command(measurements)
        if raw_command == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = raw_command
            self.candidate_frames = 1
        if self.candidate_frames >= self.confirmation_frames:
            self.command = raw_command
        context.calibrating = False
        context.command = self.command
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


class DrawLogicPointsStep:
    def process(self, context: FrameContext) -> FrameContext:
        if not context.landmarks or context.frame is None:
            return context

        height, width = context.frame.shape[:2]
        groups = (
            (MOUTH_POINTS, (0, 0, 255)),
            (EYE_CORNERS + LEFT_EYE_POINTS + RIGHT_EYE_POINTS, (255, 0, 0)),
            (LEFT_BROW_POINTS + RIGHT_BROW_POINTS, (0, 255, 255)),
        )
        for indices, color in groups:
            for index in indices:
                landmark = context.landmarks[index]
                cv2.circle(
                    context.frame,
                    (int(landmark.x * width), int(landmark.y * height)),
                    4,
                    color,
                    -1,
                )

        for start, end in ((13, 14), EYE_CORNERS):
            first, second = context.landmarks[start], context.landmarks[end]
            cv2.line(
                context.frame,
                (int(first.x * width), int(first.y * height)),
                (int(second.x * width), int(second.y * height)),
                (255, 255, 255),
                1,
            )
        if context.logic_measurements:
            metrics = context.logic_measurements
            text = (
                f"Boca Δ {metrics.get('mouth_delta', 0.0):.3f} | "
                f"Sobrancelhas Δ {metrics.get('left_brow_delta', 0.0):.3f}/"
                f"{metrics.get('right_brow_delta', 0.0):.3f}"
            )
            cv2.putText(
                context.frame,
                text,
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
        return context


class DisplayFrameStep:
    def __init__(self, title: str = "Landmarks faciais") -> None:
        self.title = title

    def process(self, context: FrameContext) -> FrameContext | None:
        if context.calibrating:
            text = "Calibrando... mantenha o rosto neutro"
            color = (0, 255, 255)
        elif context.command is not None:
            text = f"Comando: {context.command}"
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
