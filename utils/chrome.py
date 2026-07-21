import ctypes
import os
import subprocess
from time import sleep
from pathlib import Path

import pyautogui


DINO_URL = "chrome://dino/"


def focus_chrome_window() -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    window = ctypes.windll.user32.FindWindowW("Chrome_WidgetWin_1", None)
    return bool(window and ctypes.windll.user32.SetForegroundWindow(window))


def find_chrome(chrome_path: Path | None = None) -> Path:
    if chrome_path is not None:
        if chrome_path.is_file():
            return chrome_path
        raise FileNotFoundError(f"Chrome nao encontrado em: {chrome_path}")

    candidates = [
        Path(os.environ[variable]) / "Google/Chrome/Application/chrome.exe"
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
        if variable in os.environ
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Chrome nao encontrado. Informe --chrome-path.")


def launch_dino(chrome_path: Path | None = None) -> subprocess.Popen:
    chrome = find_chrome(chrome_path)
    process = subprocess.Popen([str(chrome), "--new-window", "about:blank"])
    sleep(1)
    focus_chrome_window()
    pyautogui.hotkey("ctrl", "l", _pause=False)
    pyautogui.write(DINO_URL, interval=0, _pause=False)
    pyautogui.press("enter", _pause=False)
    return process
