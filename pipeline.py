from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass
class FrameContext:
    frame: Any | None = None
    landmarks: Any | None = None
    features: list[float] | None = None
    should_exit: bool = False
    prediction: int | None = None
    action: str | None = None
    calibrating: bool = False
    logic_measurements: dict[str, float] | None = None
    logic_facts: set[str] | None = None
    inferred_action: str | None = None
    debug_step_ms: dict[str, float] | None = None
    debug_frame_ms: float | None = None
    debug_previous_step_ms: dict[str, float] | None = None
    debug_previous_frame_ms: float | None = None
    debug_average_step_ms: dict[str, float] | None = None
    debug_average_frame_ms: float | None = None
    debug_previous_average_step_ms: dict[str, float] | None = None
    debug_previous_average_frame_ms: float | None = None


class Pipeline:
    def __init__(self, steps: list[Any], profile: bool = False, clock=perf_counter) -> None:
        self.steps = steps
        self.profile = profile
        self.clock = clock
        self._previous_step_ms: dict[str, float] | None = None
        self._previous_frame_ms: float | None = None
        self._timed_frames = 0
        self._total_step_ms: dict[str, float] = {}
        self._total_frame_ms = 0.0

    def _average_step_ms(self) -> dict[str, float] | None:
        if not self._timed_frames:
            return None
        return {name: total / self._timed_frames for name, total in self._total_step_ms.items()}

    def _average_frame_ms(self) -> float | None:
        if not self._timed_frames:
            return None
        return self._total_frame_ms / self._timed_frames

    def run(self, context: FrameContext) -> FrameContext | None:
        if not self.profile:
            for step in self.steps:
                context = step.process(context)
                if context is None:
                    return None
            return context

        context.debug_previous_step_ms = self._previous_step_ms
        context.debug_previous_frame_ms = self._previous_frame_ms
        context.debug_previous_average_step_ms = self._average_step_ms()
        context.debug_previous_average_frame_ms = self._average_frame_ms()
        frame_started = self.clock()
        timings: dict[str, float] = {}
        for step in self.steps:
            step_started = self.clock()
            context = step.process(context)
            step_finished = self.clock()
            timings[type(step).__name__] = (step_finished - step_started) * 1000
            if context is None:
                return None
        context.debug_step_ms = timings
        context.debug_frame_ms = (step_finished - frame_started) * 1000
        self._timed_frames += 1
        self._total_frame_ms += context.debug_frame_ms
        for name, duration in timings.items():
            self._total_step_ms[name] = self._total_step_ms.get(name, 0.0) + duration
        context.debug_average_step_ms = self._average_step_ms()
        context.debug_average_frame_ms = self._average_frame_ms()
        self._previous_step_ms = timings.copy()
        self._previous_frame_ms = context.debug_frame_ms
        return context
