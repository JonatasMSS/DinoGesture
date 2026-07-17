"""Captura landmarks faciais em tempo real usando MediaPipe Face Mesh.

Exemplos:
    python script_face.py
    python script_face.py --print-coordinates
    python script_face.py --output face_coordinates.jsonl

Durante a execução:
    P: imprime no terminal as coordenadas do frame atual
    Q ou ESC: encerra o programa
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai coordenadas faciais da webcam com MediaPipe."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice da câmera usada pelo OpenCV (padrão: 0).",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Quantidade máxima de rostos detectados (padrão: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Arquivo JSONL opcional para salvar as coordenadas de cada frame.",
    )
    parser.add_argument(
        "--print-coordinates",
        action="store_true",
        help="Imprime continuamente as coordenadas detectadas no terminal.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Não espelha horizontalmente a imagem da câmera.",
    )
    return parser.parse_args()


def extract_coordinates(face_landmarks: Any) -> list[dict[str, float | int]]:
    """Converte os landmarks do MediaPipe em uma lista serializável."""
    return [
        {
            "index": index,
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
        }
        for index, landmark in enumerate(face_landmarks.landmark)
    ]


def draw_face_mesh(
    frame: Any,
    face_landmarks: Any,
    drawing: Any,
    drawing_styles: Any,
    face_mesh: Any,
) -> None:
    """Desenha a malha, os contornos e as íris sobre o frame."""
    drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )
    drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
    )


def build_frame_data(results: Any) -> dict[str, Any]:
    """Monta o objeto contendo as coordenadas de todos os rostos do frame."""
    faces = [
        extract_coordinates(face_landmarks)
        for face_landmarks in (results.multi_face_landmarks or [])
    ]
    return {
        "timestamp_ms": int(time.time() * 1000),
        "faces": faces,
    }


def main() -> None:
    args = parse_args()

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {args.camera}.")

    output_file = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("a", encoding="utf-8")

    current_frame_data: dict[str, Any] = {"timestamp_ms": 0, "faces": []}

    print("Câmera iniciada. Pressione P para imprimir as coordenadas.")
    print("Pressione Q ou ESC para sair.")

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=args.max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as detector:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Não foi possível ler um frame da câmera.")
                    break

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = detector.process(rgb_frame)
                rgb_frame.flags.writeable = True

                current_frame_data = build_frame_data(results)
                faces = results.multi_face_landmarks or []

                for face_landmarks in faces:
                    draw_face_mesh(
                        frame,
                        face_landmarks,
                        mp_drawing,
                        mp_drawing_styles,
                        mp_face_mesh,
                    )

                landmark_count = len(current_frame_data["faces"][0]) if faces else 0
                cv2.putText(
                    frame,
                    f"Rostos: {len(faces)} | Pontos: {landmark_count}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if faces else (0, 0, 255),
                    2,
                )

                serialized_data = None
                if faces and (args.print_coordinates or output_file):
                    serialized_data = json.dumps(
                        current_frame_data,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

                if serialized_data and args.print_coordinates:
                    print(serialized_data)

                if serialized_data and output_file:
                    output_file.write(serialized_data + "\n")
                    output_file.flush()

                cv2.imshow("MediaPipe - Landmarks faciais", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("p") and faces:
                    print(
                        json.dumps(
                            current_frame_data,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
    finally:
        if output_file:
            output_file.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
