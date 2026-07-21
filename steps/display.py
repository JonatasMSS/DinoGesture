import cv2

from pipeline import FrameContext
from .face import (
    EYE_CORNERS,
    LEFT_BROW_POINTS,
    LEFT_EYE_POINTS,
    MOUTH_POINTS,
    RIGHT_BROW_POINTS,
    RIGHT_EYE_POINTS,
)


class DrawLandmarksStep:
    def __init__(self, show_logic_points: bool = False) -> None:
        self.show_logic_points = show_logic_points

    def process(self, context: FrameContext) -> FrameContext:
        if context.landmarks:
            height, width = context.frame.shape[:2]
            for landmark in context.landmarks:
                cv2.circle(
                    context.frame,
                    (int(landmark.x * width), int(landmark.y * height)),
                    1,
                    (0, 255, 0),
                    -1,
                )
            if self.show_logic_points:
                self._draw_logic_points(context, width, height)
        return context

    @staticmethod
    def _draw_logic_points(context: FrameContext, width: int, height: int) -> None:
        groups = (
            (MOUTH_POINTS, (0, 0, 255)),
            (EYE_CORNERS + LEFT_EYE_POINTS + RIGHT_EYE_POINTS, (255, 0, 0)),
            (LEFT_BROW_POINTS + RIGHT_BROW_POINTS, (0, 255, 255)),
        )
        for indices, color in groups:
            for index in indices:
                landmark = context.landmarks[index]
                position = (int(landmark.x * width), int(landmark.y * height))
                cv2.circle(context.frame, position, 4, color, -1)
                cv2.putText(
                    context.frame,
                    str(index),
                    (position[0] + 5, position[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        for start, end in ((13, 14), EYE_CORNERS):
            first, second = context.landmarks[start], context.landmarks[end]
            cv2.line(
                context.frame,
                (int(first.x * width), int(first.y * height)),
                (int(second.x * width), int(second.y * height)),
                (255, 255, 255),
                1,
            )
        if context.logic_measurements:
            metrics = context.logic_measurements
            text = (
                f"Boca  {metrics.get('mouth_delta', 0.0):.3f} | "
                f"Sobrancelhas  {metrics.get('left_brow_delta', 0.0):.3f}/"
                f"{metrics.get('right_brow_delta', 0.0):.3f}"
            )
            cv2.putText(
                context.frame,
                text,
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )


class DisplayFrameStep:
    def __init__(self, title: str = "Landmarks faciais") -> None:
        self.title = title

    def process(self, context: FrameContext) -> FrameContext | None:
        if context.calibrating:
            text = "Calibrando... mantenha o rosto neutro"
            color = (0, 255, 255)
        elif context.command is not None:
            text = f"Comando: {context.command}"
            color = (0, 255, 0)
        else:
            text = "Sem face detectada"
            color = (0, 0, 255)
        cv2.putText(context.frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow(self.title, context.frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            context.should_exit = True
            return None
        return context
