import math
from statistics import median
from typing import Any

import cv2

from logic import KnowledgeDefinition
from model.Model import Model
from pipeline import FrameContext
from scripts.build_landmarks_csv import flatten_landmarks


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
    def __init__(self, model: Model) -> None:
        self.predictor = model

    def process(self, context: FrameContext) -> FrameContext:
        if context.features is not None:
            context.prediction = self.predictor.predict([context.features])[0]
        return context


MOUTH_POINTS = (13, 14)
EYE_CORNERS = (33, 263)
LEFT_BROW_POINTS = (70, 63, 105, 66, 107)
RIGHT_BROW_POINTS = (336, 296, 334, 293, 300)
LEFT_EYE_POINTS = (159, 145)
RIGHT_EYE_POINTS = (386, 374)


class LogicalAgentStep:
    """Percebe o rosto e consulta uma base de conhecimento proposicional.

    Args:
        knowledge: regras e limiares declarativos carregados da KB externa.
        calibration_frames: quantidade de frames neutros usados para calcular
            a mediana de referência de boca e sobrancelhas.
        confirmation_frames: quantidade de frames consecutivos necessária para
            confirmar uma decisão e reduzir oscilações causadas por ruído.
    """

    def __init__(
        self,
        knowledge: KnowledgeDefinition,
        calibration_frames: int = 24,
        confirmation_frames: int = 3,
    ) -> None:
        self.knowledge = knowledge
        self.calibration_frames = calibration_frames
        self.confirmation_frames = confirmation_frames
        self.calibration_samples: list[dict[str, float]] = []
        self.baseline: dict[str, float] | None = None
        self.action = "neutro"
        self.candidate: str | None = None
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

    def _percept_facts(self, measurements: dict[str, float]) -> set[str]:
        """Transforma medidas em fatos que serão enviados à KB com TELL."""
        assert self.baseline is not None
        mouth_delta = measurements["mouth_opening"] - self.baseline["mouth_opening"]
        left_brow_delta = measurements["left_brow_gap"] - self.baseline["left_brow_gap"]
        right_brow_delta = measurements["right_brow_gap"] - self.baseline["right_brow_gap"]
        measurements.update(
            mouth_delta=mouth_delta,
            left_brow_delta=left_brow_delta,
            right_brow_delta=right_brow_delta,
        )
        facts = {
            "boca_aberta"
            if mouth_delta >= self.knowledge.mouth_threshold
            else "boca_fechada",
            "sobrancelhas_levantadas"
            if (
                left_brow_delta >= self.knowledge.brow_threshold
                and right_brow_delta >= self.knowledge.brow_threshold
            )
            else "sobrancelhas_nao_levantadas",
        }
        return facts

    def process(self, context: FrameContext) -> FrameContext:
        context.logic_measurements = None
        context.logic_facts = None
        context.inferred_action = None
        context.calibrating = self.baseline is None
        context.action = None
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
                context.action = self.action
            return context

        facts = self._percept_facts(measurements)
        inferred_action = self.knowledge.infer_action(facts)
        context.logic_facts = facts
        context.inferred_action = inferred_action
        if inferred_action == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = inferred_action
            self.candidate_frames = 1
        if self.candidate_frames >= self.confirmation_frames:
            self.action = inferred_action
        context.calibrating = False
        context.action = self.action
        return context
