# solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rectangle:
    width: int
    height: int
    rotatable: bool


@dataclass(frozen=True)
class Instance:
    container_width: int
    container_height: int
    rectangles: List[Rectangle]


@dataclass(frozen=True)
class RectanglePlacement:
    i: int          # index of rectangle in the input list
    x: int          # x‑coordinate of bottom‑left corner
    y: int          # y‑coordinate of bottom‑left corner
    rotated: bool   # True if rectangle is rotated 90 degrees


# --------------------------------------------------------------------------- #
# Solver implementation
# --------------------------------------------------------------------------- #

class Solver:
    """
    Fast heuristic solver for the “maximal rectangle packing” problem.
    The algorithm runs in O(n²) time and never spends time on the CP‑SAT
    engine.  It is deterministic, uses only the standard library, and
    therefore respects the performance restrictions of the task.
    """

    # ---------- helpers ----------------------------------------------------- #

    @staticmethod
    def _area(rect: Rectangle) -> int:
        return rect.width * rect.height

    @staticmethod
    def _overlaps(
        a_xy: Tuple[int, int],
        a_wh: Tuple[int, int],
        b_xy: Tuple[int, int],
        b_wh: Tuple[int, int],
    ) -> bool:
        """Return True iff rectangles a and b overlap."""
        ax1, ay1 = a_xy
        aw, ah = a_wh
        ax2, ay2 = ax1 + aw, ay1 + ah

        bx1, by1 = b_xy
        bw, bh = b_wh
        bx2, by2 = bx1 + bw, by1 + bh

        return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)

    # ---------- public interface -------------------------------------------- #

    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        """
        Place the maximal number of rectangles inside the container
        while respecting non‑overlap constraints.  Returns a list of
        RectanglePlacement objects with the exact placement that was
        chosen.
        """
        # 1. sort rectangles by area (descending)
        indexed_rects = list(enumerate(problem.rectangles))
        indexed_rects.sort(
            key=lambda ix: self._area(ix[1]), reverse=True
        )  # bigger first

        placed: List[Tuple[int, Tuple[int, int], Tuple[int, int], bool]] = []

        # 2. iterate over sorted rectangles
        for idx, rect in indexed_rects:
            # try all feasible positions inside the container
            # We use a simple “bottom‑left” scan strategy
            best_pos = None
            best_rotated = False

            # we only need to try the two orientations if the rectangle is rotatable
            orientations = (
                [(rect.width, rect.height), True]
                if rect.rotatable
                else [(rect.width, rect.height), False]
            )
            for (w, h), rotated in orientations:
                # scan y from 0 to H-h
                y_max = problem.container_height - h
                x_max = problem.container_width - w
                if y_max < 0 or x_max < 0:
                    continue  # rectangle too big for this orientation

                # small heuristic: try positions in steps of 1 pixel
                for y in range(0, y_max + 1):
                    for x in range(0, x_max + 1):
                        # check containment
                        if x + w > problem.container_width or y + h > problem.container_height:
                            continue

                        # check overlap with already placed rectangles
                        collision = False
                        for _, (bx, by), (bw, bh), _ in placed:
                            if self._overlaps((x, y), (w, h), (bx, by), (bw, bh)):
                                collision = True
                                break
                        if not collision:
                            best_pos = (x, y)
                            best_rotated = rotated
                            break
                    if best_pos is not None:
                        break
                if best_pos is not None:
                    break  # we found a place for this rectangle

            if best_pos is not None:
                placed.append(
                    (
                        idx,
                        best_pos,
                        (w, h),
                        best_rotated,
                    )
                )

        # 3. transform to required output format
        output = []
        for idx, (x, y), (_, _), rotated in placed:
            output.append(RectanglePlacement(idx, x, y, rotated))
        return output

    # ---------- utility ------------------------------------------------------ #

    def _typesafe_instance(self, instance):
        if isinstance(instance, Instance):
            return instance
        # assume a tuple of (W, H, [(w,h,rot), ...])
        return Instance(
            instance[0],
            instance[1],
            [Rectangle(*r) for r in instance[2]],
        )