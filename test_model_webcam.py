"""Testa o Random Forest treinado em tempo real usando a webcam.

O fluxo permanece separado de ``main.py`` para que o teste do agente lógico
continue intacto. Saia da janela pressionando ``q`` ou ``Esc``.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from pipeline import FrameContext, Pipeline
from steps import CaptureFrameStep, DetectFaceStep, DrawLandmarksStep, MirrorFrameStep


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "random_forest.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "model" / "random_forest_metadata.joblib"

LANDMARK_COUNT = 468
NOSE_TIP = 1
RIGHT_EYE_OUTER_CORNER = 33
LEFT_EYE_OUTER_CORNER = 263
EPSILON = 1e-6

RIGHT_EYEBROW = np.array([46, 53, 52, 65, 55, 70, 63, 105, 66, 107])
LEFT_EYEBROW = np.array([276, 283, 282, 295, 285, 300, 293, 334, 296, 336])
RIGHT_EYE = np.array([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])
LEFT_EYE = np.array([263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])
RIGHT_CHEEK = np.array([50, 101, 205])
LEFT_CHEEK = np.array([280, 330, 425])
UPPER_LIP = np.array([0, 13, 37, 39, 40, 80, 81, 82, 185, 191, 267, 269, 270, 310, 311, 312, 409, 415])
LOWER_LIP = np.array([14, 17, 84, 87, 88, 91, 95, 146, 178, 181, 314, 317, 318, 321, 324, 375, 402, 405])
INNER_LIP_CONTOUR = np.array([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191])
MOUTH_VERTICAL_PAIRS = ((82, 87), (13, 14), (312, 317))
RIGHT_BROW_ENDPOINTS = (70, 107)
LEFT_BROW_ENDPOINTS = (336, 300)
RIGHT_BROW_ARCH_CENTER = 105
LEFT_BROW_ARCH_CENTER = 334
CHIN_TIP = 152


class PrepareRandomForestFeaturesStep:
    """Reproduz, de forma causal, as features criadas em data_prep.ipynb."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.coordinate_columns = list(metadata["coordinate_columns"])
        self.engineered_feature_columns = list(metadata["engineered_feature_columns"])
        self.feature_columns = list(metadata["feature_columns"])
        expected_coordinates = [
            f"{axis}_{index}"
            for index in range(LANDMARK_COUNT)
            for axis in ("x", "y", "z")
        ]
        if self.coordinate_columns != expected_coordinates:
            raise ValueError("A ordem das coordenadas nos metadados é incompatível.")
        if self.feature_columns != self.coordinate_columns + self.engineered_feature_columns:
            raise ValueError("A ordem final das features nos metadados é incompatível.")

        self.frame_index = 0
        self.last_valid_frame: int | None = None
        self.last_mouth_opening: float | None = None
        self.last_right_brow: float | None = None
        self.last_left_brow: float | None = None
        self.last_mouth_velocity: float | None = None
        self.mouth_history: deque[float] = deque(maxlen=5)
        self.right_brow_history: deque[float] = deque(maxlen=5)
        self.left_brow_history: deque[float] = deque(maxlen=5)

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second))

    @staticmethod
    def _vertical_distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(abs(first[1] - second[1]))

    @staticmethod
    def _signed_angle(first: np.ndarray, second: np.ndarray) -> float:
        difference = second - first
        return float(np.arctan2(difference[1], difference[0]))

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        x_coordinates = polygon[:, 0]
        y_coordinates = polygon[:, 1]
        return float(
            0.5
            * abs(
                np.sum(
                    x_coordinates * np.roll(y_coordinates, -1)
                    - y_coordinates * np.roll(x_coordinates, -1)
                )
            )
        )

    def _eye_aspect_ratio(
        self,
        points: np.ndarray,
        horizontal_pair: tuple[int, int],
        vertical_pairs: tuple[tuple[int, int], ...],
    ) -> float:
        horizontal = self._distance(
            points[horizontal_pair[0], :2], points[horizontal_pair[1], :2]
        )
        vertical = sum(
            self._distance(points[upper, :2], points[lower, :2])
            for upper, lower in vertical_pairs
        )
        return vertical / (len(vertical_pairs) * horizontal + EPSILON)

    @staticmethod
    def _normalize(coordinates: list[float]) -> np.ndarray:
        points = np.asarray(coordinates, dtype=np.float32).reshape(LANDMARK_COUNT, 3).copy()
        points -= points[NOSE_TIP]
        interocular_scale = np.linalg.norm(
            points[RIGHT_EYE_OUTER_CORNER, :2]
            - points[LEFT_EYE_OUTER_CORNER, :2]
        )
        if not np.isfinite(interocular_scale) or interocular_scale <= EPSILON:
            raise ValueError("Escala interocular inválida no frame atual.")
        points /= interocular_scale

        eye_direction = (
            points[LEFT_EYE_OUTER_CORNER, :2]
            - points[RIGHT_EYE_OUTER_CORNER, :2]
        )
        face_roll_radians = np.arctan2(eye_direction[1], eye_direction[0])
        cosine = np.cos(face_roll_radians)
        sine = np.sin(face_roll_radians)
        original_x = points[:, 0].copy()
        original_y = points[:, 1].copy()
        points[:, 0] = original_x * cosine + original_y * sine
        points[:, 1] = -original_x * sine + original_y * cosine
        return points

    def _temporal_features(
        self,
        frame_index: int,
        mouth_opening: float,
        right_brow: float,
        left_brow: float,
    ) -> dict[str, float]:
        if self.last_valid_frame is None:
            mouth_velocity = right_brow_velocity = left_brow_velocity = 0.0
            mouth_acceleration = 0.0
        else:
            frame_step = frame_index - self.last_valid_frame
            if frame_step <= 0:
                raise ValueError("A sequência temporal dos frames é inválida.")
            mouth_velocity = (mouth_opening - self.last_mouth_opening) / frame_step
            right_brow_velocity = (right_brow - self.last_right_brow) / frame_step
            left_brow_velocity = (left_brow - self.last_left_brow) / frame_step
            mouth_acceleration = (
                (mouth_velocity - self.last_mouth_velocity) / frame_step
                if self.last_mouth_velocity is not None
                else 0.0
            )

        self.mouth_history.append(mouth_opening)
        self.right_brow_history.append(right_brow)
        self.left_brow_history.append(left_brow)

        self.last_valid_frame = frame_index
        self.last_mouth_opening = mouth_opening
        self.last_right_brow = right_brow
        self.last_left_brow = left_brow
        self.last_mouth_velocity = mouth_velocity

        return {
            "velocidade_abertura_boca": mouth_velocity,
            "aceleracao_abertura_boca": mouth_acceleration,
            "velocidade_sobrancelha_direita": right_brow_velocity,
            "velocidade_sobrancelha_esquerda": left_brow_velocity,
            "media_movel_5_abertura_boca": float(np.mean(self.mouth_history)),
            "media_movel_5_sobrancelha_direita": float(np.mean(self.right_brow_history)),
            "media_movel_5_sobrancelha_esquerda": float(np.mean(self.left_brow_history)),
        }

    def transform(self, coordinates: list[float], frame_index: int) -> list[float]:
        points = self._normalize(coordinates)

        right_eyebrow_center = points[RIGHT_EYEBROW, :2].mean(axis=0)
        left_eyebrow_center = points[LEFT_EYEBROW, :2].mean(axis=0)
        eyebrows_center = (right_eyebrow_center + left_eyebrow_center) / 2
        right_eye_center = points[RIGHT_EYE, :2].mean(axis=0)
        left_eye_center = points[LEFT_EYE, :2].mean(axis=0)
        right_cheek_center = points[RIGHT_CHEEK, :2].mean(axis=0)
        left_cheek_center = points[LEFT_CHEEK, :2].mean(axis=0)
        cheeks_center = (right_cheek_center + left_cheek_center) / 2

        top_face_point = points[np.argmin(points[:, 1]), :2]
        upper_lip_region = points[UPPER_LIP, :2]
        lower_lip_region = points[LOWER_LIP, :2]
        top_upper_lip_point = upper_lip_region[np.argmin(upper_lip_region[:, 1])]
        bottom_lower_lip_point = lower_lip_region[np.argmax(lower_lip_region[:, 1])]

        right_eyebrow_eye_distance = self._distance(right_eyebrow_center, right_eye_center)
        left_eyebrow_eye_distance = self._distance(left_eyebrow_center, left_eye_center)
        mouth_vertical_height = self._vertical_distance(points[13, :2], points[14, :2])
        mouth_width = self._distance(points[61, :2], points[291, :2])
        mouth_average_height = float(
            np.mean([
                self._vertical_distance(points[upper, :2], points[lower, :2])
                for upper, lower in MOUTH_VERTICAL_PAIRS
            ])
        )
        right_brow_vertical = self._vertical_distance(right_eye_center, right_eyebrow_center)
        left_brow_vertical = self._vertical_distance(left_eye_center, left_eyebrow_center)
        right_eye_aspect_ratio = self._eye_aspect_ratio(
            points, (33, 133), ((159, 145), (158, 153))
        )
        left_eye_aspect_ratio = self._eye_aspect_ratio(
            points, (362, 263), ((386, 374), (385, 380))
        )

        engineered = {
            "dist_centro_sobrancelhas_centro_bochechas": self._distance(eyebrows_center, cheeks_center),
            "dist_centro_sobrancelhas_topo_rosto": self._distance(eyebrows_center, top_face_point),
            "dist_sobrancelha_direita_olho_direito": right_eyebrow_eye_distance,
            "dist_sobrancelha_esquerda_olho_esquerdo": left_eyebrow_eye_distance,
            "dist_media_sobrancelhas_olhos": (right_eyebrow_eye_distance + left_eyebrow_eye_distance) / 2,
            "dist_topo_labio_superior_base_labio_inferior": self._distance(top_upper_lip_point, bottom_lower_lip_point),
            "dist_topo_labio_superior_ponta_queixo": self._distance(top_upper_lip_point, points[CHIN_TIP, :2]),
            "altura_vertical_boca": mouth_vertical_height,
            "largura_boca": mouth_width,
            "altura_media_boca": mouth_average_height,
            "mouth_aspect_ratio": mouth_average_height / (mouth_width + EPSILON),
            "area_abertura_boca": self._polygon_area(points[INNER_LIP_CONTOUR, :2]),
            "dist_vertical_sobrancelha_direita_olho": right_brow_vertical,
            "dist_vertical_sobrancelha_esquerda_olho": left_brow_vertical,
            "dist_vertical_media_sobrancelhas_olhos": (right_brow_vertical + left_brow_vertical) / 2,
            "assimetria_sobrancelhas": abs(right_brow_vertical - left_brow_vertical),
            "inclinacao_sobrancelha_direita": self._signed_angle(points[RIGHT_BROW_ENDPOINTS[0], :2], points[RIGHT_BROW_ENDPOINTS[1], :2]),
            "inclinacao_sobrancelha_esquerda": self._signed_angle(points[LEFT_BROW_ENDPOINTS[0], :2], points[LEFT_BROW_ENDPOINTS[1], :2]),
            "curvatura_sobrancelha_direita": float(points[list(RIGHT_BROW_ENDPOINTS), 1].mean() - points[RIGHT_BROW_ARCH_CENTER, 1]),
            "curvatura_sobrancelha_esquerda": float(points[list(LEFT_BROW_ENDPOINTS), 1].mean() - points[LEFT_BROW_ARCH_CENTER, 1]),
            "eye_aspect_ratio_direito": right_eye_aspect_ratio,
            "eye_aspect_ratio_esquerdo": left_eye_aspect_ratio,
            "eye_aspect_ratio_medio": (right_eye_aspect_ratio + left_eye_aspect_ratio) / 2,
        }
        engineered.update(
            self._temporal_features(
                frame_index,
                mouth_average_height,
                right_brow_vertical,
                left_brow_vertical,
            )
        )
        return points.reshape(-1).tolist() + [
            float(engineered[name]) for name in self.engineered_feature_columns
        ]

    def process(self, context: FrameContext) -> FrameContext:
        self.frame_index += 1
        if context.features is not None:
            context.features = self.transform(context.features, self.frame_index)
        return context


class PredictRandomForestStep:
    def __init__(self, model: Any, feature_columns: list[str]) -> None:
        self.model = model
        self.feature_columns = feature_columns

    def process(self, context: FrameContext) -> FrameContext:
        if context.features is None:
            context.prediction = None
            return context
        sample = pd.DataFrame([context.features], columns=self.feature_columns)
        context.prediction = self.model.predict(sample)[0]
        return context


class DisplayModelPredictionStep:
    def __init__(self, title: str = "Teste do Random Forest") -> None:
        self.title = title

    def process(self, context: FrameContext) -> FrameContext | None:
        if context.prediction is None:
            text = "Sem face detectada"
            color = (0, 0, 255)
        else:
            label = str(context.prediction).replace("_", " ")
            text = f"Predicao: {label}"
            color = (0, 255, 0)
        cv2.putText(
            context.frame,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        cv2.imshow(self.title, context.frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            context.should_exit = True
            return None
        return context


def load_artifacts(model_path: Path, metadata_path: Path) -> tuple[Any, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    feature_columns = list(metadata["feature_columns"])
    if model.n_features_in_ != len(feature_columns):
        raise ValueError("O modelo e os metadados têm quantidades de features diferentes.")
    if hasattr(model, "feature_names_in_") and model.feature_names_in_.tolist() != feature_columns:
        raise ValueError("A ordem das features do modelo difere dos metadados.")
    return model, metadata


def camera_index(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("O índice da câmera não pode ser negativo.")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testa o Random Forest treinado em frames da webcam."
    )
    parser.add_argument("--camera", type=camera_index, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    model, metadata = load_artifacts(args.model.resolve(), args.metadata.resolve())
    feature_step = PrepareRandomForestFeaturesStep(metadata)
    print(f"Modelo carregado: {args.model.resolve()}")
    print(f"Classes: {', '.join(map(str, model.classes_))}")
    print("Pressione q ou Esc para encerrar.")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {args.camera}.")

    face_mesh = mp.solutions.face_mesh
    try:
        with face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
        ) as detector:
            pipeline = Pipeline([
                CaptureFrameStep(camera),
                MirrorFrameStep(),
                DetectFaceStep(detector),
                feature_step,
                PredictRandomForestStep(model, feature_step.feature_columns),
                DrawLandmarksStep(),
                DisplayModelPredictionStep(),
            ])
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
