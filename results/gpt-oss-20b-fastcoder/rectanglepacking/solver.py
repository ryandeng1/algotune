from typing import NamedTuple, List


class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool


class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: List[Rectangle]


class RectanglePlacement(NamedTuple):
    i: int
    x: int
    y: int
    rotated: bool


class Solver:
    """Fast heuristic solver for the rectangle knapsack problem."""

    def solve(self, problem: Instance, time_limit: float = 900.0) -> List[RectanglePlacement]:
        """Place as many rectangles as possible using a simple skyline algorithm."""
        # Sort rectangles by area (descending) to try larger ones first
        rects = sorted(
            [(i, r) for i, r in enumerate(problem.rectangles)],
            key=lambda ir: ir[1].width * ir[1].height,
            reverse=True,
        )

        placed: List[RectanglePlacement] = []
        used_heights = [0] * problem.container_width

        for i, rect in rects:
            width = rect.width
            height = rect.height
            best_pos = None
            best_y = None

            # Try original orientation
            for rotated in (False, True) if rect.rotatable else (False,):
                w, h = (height, width) if rotated else (width, height)
                if w > problem.container_width or h > problem.container_height:
                    continue

                # Find leftmost position where the rectangle fits
                for x in range(0, problem.container_width - w + 1):
                    # Find max height in this span
                    max_y = max(used_heights[x : x + w])
                    if max_y + h <= problem.container_height:
                        if best_pos is None or max_y < best_y:
                            best_pos = (x, max_y, rotated, w, h)
                            best_y = max_y
                            # early exit if bottom-left
                            if max_y == 0:
                                break
                if best_pos is not None:
                    break

            if best_pos is not None:
                x, y, rotated, w, h = best_pos
                # Update skyline
                for idx in range(x, x + w):
                    used_heights[idx] = y + h
                placed.append(RectanglePlacement(i, x, y, rotated))

        return placed

    # Helper to create an Instance if needed
    def _typesafe_instance(self, instance):
        if isinstance(instance, Instance):
            return instance
        return Instance(
            instance[0], instance[1],
            [Rectangle(*r) for r in instance[2]]
        )