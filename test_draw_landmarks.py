import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import FrameContext
from steps import DrawLandmarksStep


class DrawLandmarksStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = FrameContext(
            frame=SimpleNamespace(shape=(100, 200, 3)),
            landmarks=[SimpleNamespace(x=0.5, y=0.5) for _ in range(468)],
            logic_measurements={"mouth_delta": 0.1},
        )

    def test_logic_points_are_drawn_only_when_enabled(self) -> None:
        with (
            patch("steps.display.cv2.circle") as circle,
            patch("steps.display.cv2.line") as line,
            patch("steps.display.cv2.putText") as text,
        ):
            DrawLandmarksStep(show_logic_points=True).process(self.context)

        self.assertEqual(circle.call_count, 486)
        self.assertEqual(line.call_count, 2)
        self.assertEqual(text.call_count, 19)
        self.assertTrue(any(call.args[1] == "33" for call in text.call_args_list))
        self.assertTrue(any(call.args[1] == "263" for call in text.call_args_list))


if __name__ == "__main__":
    unittest.main()
