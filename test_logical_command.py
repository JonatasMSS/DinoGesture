import unittest
from types import SimpleNamespace

from pipeline import FrameContext
from steps import LogicalCommandStep


def landmarks(mouth_open: bool = False, brows_raised: bool = False):
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(468)]
    points[33] = SimpleNamespace(x=0.3, y=0.4, z=0.0)
    points[263] = SimpleNamespace(x=0.7, y=0.4, z=0.0)
    points[13] = SimpleNamespace(x=0.5, y=0.55, z=0.0)
    points[14] = SimpleNamespace(x=0.5, y=0.65 if mouth_open else 0.57, z=0.0)
    for index in (159, 145, 386, 374):
        points[index] = SimpleNamespace(x=0.5, y=0.4, z=0.0)
    brow_y = 0.25 if brows_raised else 0.3
    for index in (70, 63, 105, 66, 107, 336, 296, 334, 293, 300):
        points[index] = SimpleNamespace(x=0.5, y=brow_y, z=0.0)
    return points


class LogicalCommandStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.step = LogicalCommandStep(calibration_frames=3, confirmation_frames=3)
        for _ in range(3):
            self.context = self.step.process(FrameContext(landmarks=landmarks()))

    def command_after_confirmation(self, **expression) -> int:
        for _ in range(3):
            self.context = self.step.process(FrameContext(landmarks=landmarks(**expression)))
        return self.context.command

    def test_calibrates_and_returns_neutral_command(self) -> None:
        self.assertFalse(self.context.calibrating)
        self.assertEqual(self.context.command, 0)

    def test_detects_mouth_open_with_priority(self) -> None:
        self.assertEqual(self.command_after_confirmation(mouth_open=True), 1)
        self.setUp()
        self.assertEqual(self.command_after_confirmation(mouth_open=True, brows_raised=True), 1)

    def test_detects_both_raised_eyebrows(self) -> None:
        self.assertEqual(self.command_after_confirmation(brows_raised=True), 2)

    def test_no_face_has_no_command(self) -> None:
        context = self.step.process(FrameContext(landmarks=None))
        self.assertIsNone(context.command)


if __name__ == "__main__":
    unittest.main()
