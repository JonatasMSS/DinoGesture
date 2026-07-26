"""Captura uma sessão de expressão facial para criar um dataset."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2


FPS = 24
PREPARATION_SECONDS = 3
FRAME_COUNT = 120
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_ROOT = PROJECT_ROOT / "data" / "recordings"
INVALID_FOLDER_CHARACTERS = re.compile(r'[<>:"/\\|?*]')


def expression_name(value: str) -> str:
    """Valida o nome que será usado como pasta e label da expressão."""
    value = value.strip()
    if not value or value in {".", ".."} or value.endswith((".", " ")):
        raise ValueError("O nome da expressão é inválido.")
    if INVALID_FOLDER_CHARACTERS.search(value):
        raise ValueError("O nome contém caracteres inválidos para uma pasta.")
    return value


def new_session_id() -> str:
    return f"{datetime.now():%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"


def save_session(pending_session: Path, recordings_root: Path, label: str) -> Path:
    """Move uma sessão temporária para a pasta da expressão informada."""
    label = expression_name(label)
    destination_parent = recordings_root / label
    destination_parent.mkdir(parents=True, exist_ok=True)

    destination = destination_parent / new_session_id()
    while destination.exists():
        destination = destination_parent / new_session_id()
    shutil.move(str(pending_session), str(destination))
    return destination


def draw_status(frame, text: str, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )
    cv2.putText(
        frame,
        "Q ou ESC: cancelar",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def save_png(output: Path, frame) -> None:
    """Salva um frame PNG, inclusive em caminhos Unicode no Windows."""
    encoded, buffer = cv2.imencode(".png", frame)
    if not encoded:
        raise RuntimeError(f"Não foi possível codificar {output.name} como PNG.")

    try:
        output.write_bytes(buffer.tobytes())
    except OSError as error:
        raise RuntimeError(f"Não foi possível salvar {output.name}: {error}") from error


def prompt_label() -> str:
    while True:
        try:
            return expression_name(input("Nome da expressão: "))
        except ValueError as error:
            print(error)


def capture(camera_index: int, recordings_root: Path = RECORDINGS_ROOT) -> Path | None:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {camera_index}.")

    camera.set(cv2.CAP_PROP_FPS, FPS)
    pending_session = recordings_root / ".pending" / new_session_id()
    pending_session.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    recording_started_at: float | None = None
    next_frame_at: float | None = None
    saved_frames = 0
    cancelled = False

    try:
        while saved_frames < FRAME_COUNT:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Não foi possível ler um frame da câmera.")

            elapsed = time.monotonic() - started_at
            if elapsed < PREPARATION_SECONDS:
                remaining = max(1, math.ceil(PREPARATION_SECONDS - elapsed))
                draw_status(frame, f"Prepare-se: {remaining}", (0, 255, 255))
            else:
                if recording_started_at is None:
                    recording_started_at = time.monotonic()
                    next_frame_at = recording_started_at

                now = time.monotonic()
                if now >= next_frame_at:
                    output = pending_session / f"frame_{saved_frames + 1:04d}.png"
                    save_png(output, frame)
                    saved_frames += 1
                    next_frame_at += 1 / FPS
                draw_status(frame, f"Gravando: {saved_frames}/{FRAME_COUNT}", (0, 0, 255))

            cv2.imshow("Coleta de expressão", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cancelled = True
                break
    except Exception:
        shutil.rmtree(pending_session, ignore_errors=True)
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if cancelled:
        shutil.rmtree(pending_session, ignore_errors=True)
        print("Coleta cancelada.")
        return None

    elapsed = time.monotonic() - (recording_started_at or time.monotonic())
    print(f"{saved_frames} frames capturados em {elapsed:.2f} s.")
    try:
        destination = save_session(pending_session, recordings_root, prompt_label())
    except (EOFError, KeyboardInterrupt):
        shutil.rmtree(pending_session, ignore_errors=True)
        print("Coleta descartada.")
        return None

    print(f"Sessão salva em: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta frames de uma expressão facial.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera.")
    args = parser.parse_args()
    capture(args.camera)


if __name__ == "__main__":
    main()
