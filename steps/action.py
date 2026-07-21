from time import monotonic

import pyautogui

from pipeline import FrameContext
from utils.chrome import focus_chrome_window


VK_SPACE = "space"
VK_DOWN = "down"
ACTION_JUMP = "pular"
ACTION_DUCK = "abaixar"


def tap_key(key: str) -> None:
    """Envia um toque sem a pausa padrão do PyAutoGUI."""
    pyautogui.press(key, _pause=False)


class DinoActionStep:
    """Converte ações inferidas pela KB em teclas para o Chrome Dino."""

    def __init__(self, debug_mode: bool = False, clock=monotonic) -> None:
        self.debug_mode = debug_mode
        self.clock = clock
        self.ready_at: float | None = None
        self.down_pressed = False
        self.started = False
        self.last_action: str | None = None
        self.ready_announced = False

    def _tap(self, key: str) -> None:
        if self.debug_mode:
            focus_chrome_window()
        tap_key(key)

    def _hold_down(self) -> None:
        if self.down_pressed:
            return
        if self.debug_mode:
            focus_chrome_window()
        pyautogui.keyDown(VK_DOWN, _pause=False)
        self.down_pressed = True

    def _release_down(self) -> None:
        if not self.down_pressed:
            return
        if self.debug_mode:
            focus_chrome_window()
        pyautogui.keyUp(VK_DOWN, _pause=False)
        self.down_pressed = False

    def process(self, context: FrameContext) -> FrameContext:
        now = self.clock()
        if context.calibrating:
            self._release_down()
            self.ready_at = None
            self.started = False
            self.last_action = None
            self.ready_announced = False
            return context

        if self.ready_at is None:
            self.ready_at = now + 2
            print("Calibracao concluida. Aguarde 2 segundos...")
            return context
        if now < self.ready_at:
            return context
        if not self.ready_announced:
            print("Pronto. Levante as sobrancelhas para iniciar o jogo.")
            self.ready_announced = True

        if not self.started:
            if context.action == ACTION_JUMP:
                self._tap(VK_SPACE)
                self.started = True
                self.last_action = ACTION_JUMP
            return context

        if context.action != self.last_action:
            if context.action == ACTION_JUMP:
                self._release_down()
                self._tap(VK_SPACE)
            elif context.action == ACTION_DUCK:
                self._hold_down()
            else:
                self._release_down()
            self.last_action = context.action
        return context
