"""Abre e controla o jogo do dinossauro do Google Chrome.

O modulo nao depende da pipeline principal. Para uma integracao futura, crie uma
instancia de :class:`DinoGameController` e chame ``press_up()``, ``press_down()``
ou ``execute_action()`` de acordo com o gesto detectado.

Execucao independente::

    uv run python scripts/script_dinogame.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import keyboard
import pyautogui
import pyperclip


DINO_URL = "chrome://dino/"
DEFAULT_STARTUP_TIMEOUT = 10.0
DEFAULT_DUCK_DURATION = 0.35
DEFAULT_JUMP_COOLDOWN = 0.35

GameAction = Literal["up", "down", "down_start", "down_end"]


def find_chrome_executable() -> Path:
    """Localiza o executavel do Google Chrome no Windows."""
    executable = shutil.which("chrome") or shutil.which("chrome.exe")
    if executable:
        return Path(executable)

    candidate_roots = (
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    )
    for root in filter(None, candidate_roots):
        candidate = Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Google Chrome nao encontrado. Informe o caminho com --chrome-path."
    )


def _window_id(window: object) -> int | None:
    """Retorna o identificador nativo da janela usado pelo PyGetWindow."""
    return getattr(window, "_hWnd", None)


def _is_chrome_window(window: object) -> bool:
    title = str(getattr(window, "title", "")).casefold()
    # Nao procure apenas por "dino": o nome do projeto DinoGesture tambem
    # aparece no titulo do VS Code e faria os comandos irem para o editor.
    return "google chrome" in title


class DinoGameController:
    """Controla uma janela dedicada do Chrome usando as setas do teclado.

    A classe mantem a abertura do navegador separada dos comandos. Isso permite
    instancia-la uma vez na inicializacao da pipeline e reutiliza-la a cada frame.
    """

    def __init__(
        self,
        chrome_path: str | Path | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        focus_delay: float = 0.1,
        jump_cooldown: float = DEFAULT_JUMP_COOLDOWN,
    ) -> None:
        if jump_cooldown < 0:
            raise ValueError("O intervalo entre pulos deve ser maior ou igual a zero.")

        self.chrome_path = Path(chrome_path) if chrome_path else None
        self.startup_timeout = startup_timeout
        self.focus_delay = focus_delay
        self.jump_cooldown = jump_cooldown
        self.process: subprocess.Popen[bytes] | None = None
        self.window: object | None = None
        self._down_is_held = False
        self._jump_was_detected = False
        self._last_jump_time = float("-inf")

    @property
    def is_open(self) -> bool:
        """Indica se a janela encontrada ainda parece estar aberta."""
        return self.window is not None and _window_id(self.window) in {
            _window_id(window) for window in pyautogui.getAllWindows()
        }

    def open(self) -> DinoGameController:
        """Abre ``chrome://dino/`` em uma nova janela e aguarda ela aparecer."""
        if self.is_open:
            self.focus()
            return self

        chrome = self.chrome_path or find_chrome_executable()
        if not chrome.is_file():
            raise FileNotFoundError(f"Executavel do Chrome nao encontrado: {chrome}")

        previous_window_ids = {
            _window_id(window) for window in pyautogui.getAllWindows()
        }
        self.process = subprocess.Popen(
            # O Chrome pode ignorar paginas chrome:// recebidas pela linha de
            # comando. Primeiro abrimos uma pagina comum e navegamos pela barra.
            [str(chrome), "--new-window", "about:blank"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.window = self._wait_for_window(previous_window_ids)
        self.focus()
        self.navigate_to_game()
        return self

    def navigate_to_game(self) -> None:
        """Abre o Dino pela barra de enderecos da janela controlada."""
        self.focus()
        # Colar evita que ':' e '/' sejam digitados incorretamente em teclados
        # ABNT2. O texto anterior da area de transferencia e preservado.
        previous_clipboard = pyperclip.paste()
        try:
            pyperclip.copy(DINO_URL)
            pyautogui.hotkey("ctrl", "l")
            pyautogui.hotkey("ctrl", "v")
            pyautogui.press("enter")
        finally:
            pyperclip.copy(previous_clipboard)
        time.sleep(1.0)

    def _wait_for_window(self, previous_window_ids: set[int | None]) -> object:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            windows = pyautogui.getAllWindows()
            new_chrome_windows = [
                window
                for window in windows
                if _window_id(window) not in previous_window_ids
                and _is_chrome_window(window)
            ]
            if new_chrome_windows:
                return new_chrome_windows[-1]

            # Em algumas configuracoes o Chrome reutiliza uma janela existente.
            matching_windows = [window for window in windows if _is_chrome_window(window)]
            if matching_windows:
                dino_windows = [
                    window
                    for window in matching_windows
                    if any(
                        word in str(getattr(window, "title", "")).casefold()
                        for word in ("dino", "dinosaur", "dinossauro")
                    )
                ]
                if dino_windows:
                    return dino_windows[-1]

            time.sleep(0.1)

        raise TimeoutError(
            "O Chrome foi iniciado, mas a janela do jogo nao foi encontrada."
        )

    def focus(self) -> None:
        """Traz a janela do jogo para frente antes de enviar um comando."""
        if self.window is None:
            raise RuntimeError("O jogo ainda nao foi aberto. Chame open() primeiro.")

        if getattr(self.window, "isMinimized", False):
            self.window.restore()
        self.window.activate()
        time.sleep(self.focus_delay)

    def press_up(self) -> None:
        """Pressiona a seta para cima para iniciar o jogo ou pular."""
        self.focus()
        pyautogui.press("up")

    def press_down(self, duration: float = DEFAULT_DUCK_DURATION) -> None:
        """Mantem a seta para baixo pressionada pelo tempo informado."""
        if duration < 0:
            raise ValueError("A duracao deve ser maior ou igual a zero.")

        self.focus()
        pyautogui.keyDown("down")
        try:
            time.sleep(duration)
        finally:
            pyautogui.keyUp("down")
            self._down_is_held = False

    def start_down(self) -> None:
        """Inicia o agachamento e mantem a seta para baixo pressionada."""
        if self._down_is_held:
            return
        self.focus()
        pyautogui.keyDown("down")
        self._down_is_held = True

    def stop_down(self) -> None:
        """Solta a seta para baixo apos um agachamento continuo."""
        if not self._down_is_held:
            return
        pyautogui.keyUp("down")
        self._down_is_held = False

    def execute_action(
        self,
        action: GameAction,
        down_duration: float = DEFAULT_DUCK_DURATION,
    ) -> None:
        """Executa uma acao textual, facilitando a chamada pela pipeline."""
        actions = {
            "up": self.press_up,
            "down": lambda: self.press_down(down_duration),
            "down_start": self.start_down,
            "down_end": self.stop_down,
        }
        try:
            command = actions[action]
        except KeyError as error:
            raise ValueError(f"Acao do jogo desconhecida: {action}") from error
        command()

    def update(self, jump_detected: bool, duck_detected: bool) -> None:
        """Atualiza o controle com o resultado de um frame da pipeline.

        O pulo acontece somente na transicao de ``False`` para ``True`` e
        respeita ``jump_cooldown``. O agachamento permanece ativo enquanto
        ``duck_detected`` for verdadeiro, sem bloquear o processamento de frames.
        Quando os dois gestos aparecem juntos, o pulo tem prioridade.
        """
        now = time.monotonic()
        jump_started = jump_detected and not self._jump_was_detected

        if jump_detected:
            self.stop_down()
            if jump_started and now - self._last_jump_time >= self.jump_cooldown:
                self.press_up()
                self._last_jump_time = now
        elif duck_detected:
            self.start_down()
        else:
            self.stop_down()

        self._jump_was_detected = jump_detected

    def reset_state(self) -> None:
        """Libera comandos e reinicia o estado entre sessoes da pipeline."""
        self.stop_down()
        self._jump_was_detected = False
        self._last_jump_time = float("-inf")

    def close(self) -> None:
        """Solta teclas pendentes e fecha somente a janela controlada."""
        self.reset_state()
        if self.window is not None and self.is_open:
            self.window.close()
        self.window = None


def open_dino_game(
    chrome_path: str | Path | None = None,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
) -> DinoGameController:
    """Cria o controlador e abre o jogo; ponto de entrada para integracoes."""
    return DinoGameController(chrome_path, startup_timeout).open()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Abre o Chrome Dino e controla o jogo pelo terminal."
    )
    parser.add_argument(
        "--chrome-path",
        type=Path,
        help="Caminho opcional para chrome.exe.",
    )
    return parser.parse_args()


def main() -> None:
    """Oferece atalhos globais para testar sem usar a pipeline principal."""
    args = parse_args()
    controller = open_dino_game(args.chrome_path)

    print("Chrome Dino aberto.")
    print("U pula | segure D para abaixar | Q encerra o controle.")
    print("Os atalhos funcionam enquanto a janela do Chrome estiver ativa.")

    hotkeys = [
        keyboard.add_hotkey(
            "u",
            controller.press_up,
            suppress=True,
            trigger_on_release=True,
        ),
        keyboard.on_press_key(
            "d",
            lambda _: controller.start_down(),
            suppress=True,
        ),
        keyboard.on_release_key(
            "d",
            lambda _: controller.stop_down(),
            suppress=True,
        ),
    ]
    try:
        keyboard.wait("q", suppress=True)
    finally:
        for hotkey in hotkeys:
            keyboard.unhook(hotkey)
        controller.reset_state()


if __name__ == "__main__":
    main()
