import unittest
from unittest.mock import Mock, patch

from pipeline import FrameContext
from steps.action import DinoActionStep, VK_DOWN, VK_SPACE, tap_key


class DinoActionStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Mock(return_value=100.0)
        self.step = DinoActionStep(clock=self.clock)

    def process(self, command: int | None, calibrating: bool = False) -> None:
        self.step.process(FrameContext(command=command, calibrating=calibrating))

    @patch("steps.action.tap_key")
    def test_waits_for_calibration_and_two_seconds(self, tap_key: Mock) -> None:
        self.process(2, calibrating=True)
        self.process(2)
        self.clock.return_value = 101.9
        self.process(2)
        tap_key.assert_not_called()

        self.clock.return_value = 102.0
        self.process(2)
        tap_key.assert_called_once_with(VK_SPACE)

    @patch("steps.action.pyautogui.keyUp")
    @patch("steps.action.pyautogui.keyDown")
    @patch("steps.action.tap_key")
    def test_sends_actions_once_per_command_transition(
        self, tap_key: Mock, key_down: Mock, key_up: Mock
    ) -> None:
        self.process(0)
        self.clock.return_value = 102.0
        self.process(2)
        self.process(2)
        self.process(0)
        self.process(1)
        self.process(1)

        tap_key.assert_called_once_with(VK_SPACE)
        key_down.assert_called_once_with(VK_DOWN, _pause=False)
        key_up.assert_not_called()

        self.process(0)
        key_up.assert_called_once_with(VK_DOWN, _pause=False)

    @patch("steps.action.tap_key")
    @patch("steps.action.focus_chrome_window")
    def test_debug_mode_focuses_chrome_before_action(
        self, focus_chrome_window: Mock, tap_key: Mock
    ) -> None:
        step = DinoActionStep(debug_mode=True, clock=Mock(side_effect=(0.0, 2.0)))
        step.process(FrameContext(command=0))
        step.process(FrameContext(command=2))

        focus_chrome_window.assert_called_once_with()
        tap_key.assert_called_once_with(VK_SPACE)

    @patch("steps.action.pyautogui.press")
    def test_tap_key_uses_pyautogui_without_pause(self, press: Mock) -> None:
        tap_key(VK_SPACE)

        press.assert_called_once_with(VK_SPACE, _pause=False)


if __name__ == "__main__":
    unittest.main()
