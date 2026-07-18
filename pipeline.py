from dataclasses import dataclass
from typing import Any


@dataclass
class FrameContext:
    frame: Any | None = None
    landmarks: Any | None = None
    should_exit: bool = False


class Pipeline:
    def __init__(self, steps: list[Any]) -> None:
        self.steps = steps

    def run(self, context: FrameContext) -> FrameContext | None:
        for step in self.steps:
            context = step.process(context)
            if context is None:
                return None
        return context
