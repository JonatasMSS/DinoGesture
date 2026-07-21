import unittest
from types import SimpleNamespace

from main import build_steps, validate_args
from steps import DinoActionStep, DisplayFrameStep, DrawLandmarksStep


class GamePipelineTest(unittest.TestCase):
    def args(self, game_control: bool, debug_mode: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            game_control=game_control,
            debug_mode=debug_mode,
            mouth_threshold=0.08,
            brow_threshold=0.04,
            show_logic_points=False,
        )

    def test_game_mode_omits_visual_steps_without_debug(self) -> None:
        steps = build_steps(self.args(game_control=True), object(), object())

        self.assertTrue(any(isinstance(step, DinoActionStep) for step in steps))
        self.assertFalse(any(isinstance(step, DrawLandmarksStep) for step in steps))
        self.assertFalse(any(isinstance(step, DisplayFrameStep) for step in steps))

    def test_debug_mode_includes_visual_steps(self) -> None:
        steps = build_steps(self.args(game_control=True, debug_mode=True), object(), object())

        drawing = next(step for step in steps if isinstance(step, DrawLandmarksStep))
        self.assertTrue(drawing.show_logic_points)
        self.assertTrue(any(isinstance(step, DisplayFrameStep) for step in steps))

    def test_debug_mode_requires_game_control(self) -> None:
        with self.assertRaises(ValueError):
            validate_args(self.args(game_control=False, debug_mode=True))


if __name__ == "__main__":
    unittest.main()
