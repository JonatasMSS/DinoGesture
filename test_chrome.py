import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from utils.chrome import DINO_URL, launch_dino


class ChromeTest(unittest.TestCase):
    @patch("utils.chrome.pyautogui.press")
    @patch("utils.chrome.pyautogui.write")
    @patch("utils.chrome.pyautogui.hotkey")
    @patch("utils.chrome.focus_chrome_window", return_value=True)
    @patch("utils.chrome.sleep")
    @patch("utils.chrome.subprocess.Popen")
    @patch("utils.chrome.find_chrome", return_value=Path("C:/Chrome/chrome.exe"))
    def test_launches_local_dino_page(
        self,
        find_chrome: Mock,
        popen: Mock,
        sleep: Mock,
        focus_chrome_window: Mock,
        hotkey: Mock,
        write: Mock,
        press: Mock,
    ) -> None:
        requested_path = Path("C:/custom/chrome.exe")

        launch_dino(requested_path)

        find_chrome.assert_called_once_with(requested_path)
        popen.assert_called_once_with(
            [str(Path("C:/Chrome/chrome.exe")), "--new-window", "about:blank"]
        )
        sleep.assert_called_once_with(1)
        focus_chrome_window.assert_called_once_with()
        hotkey.assert_called_once_with("ctrl", "l", _pause=False)
        write.assert_called_once_with(DINO_URL, interval=0, _pause=False)
        press.assert_called_once_with("enter", _pause=False)

    @patch("utils.chrome.pyautogui.press")
    @patch("utils.chrome.pyautogui.write")
    @patch("utils.chrome.pyautogui.hotkey")
    @patch("utils.chrome.focus_chrome_window", return_value=False)
    @patch("utils.chrome.sleep")
    @patch("utils.chrome.subprocess.Popen")
    @patch("utils.chrome.find_chrome", return_value=Path("C:/Chrome/chrome.exe"))
    def test_navigates_even_when_windows_focus_is_unavailable(
        self,
        _find_chrome: Mock,
        _popen: Mock,
        _sleep: Mock,
        _focus_chrome_window: Mock,
        hotkey: Mock,
        _write: Mock,
        _press: Mock,
    ) -> None:
        launch_dino()

        hotkey.assert_called_once_with("ctrl", "l", _pause=False)


if __name__ == "__main__":
    unittest.main()
