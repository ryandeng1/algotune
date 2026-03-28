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
    """
    Very fast but sub‑optimal heuristic: pack rectangles in order
    of decreasing area; try the original orientation first, then
    the rotated one (if allowed).  Whenever a rectangle fits in the
    next free cell (scanned left‑to‑right, bottom‑to‑top) it is
    placed there.  This algorithm runs in O(n^2) time and uses only
    integer arithmetic, so it stays well below any time limit.
    """

    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        # Write the instance as a typesafe Instance
        container_w, container_h, rects = problem.container_width, problem.container_height, problem.rectangles
        # Sort rectangles by area descending, keep original index
        indexed = sorted(
            [(i, r.width * r.height, r) for i, r in enumerate(rects)],
            key=lambda t: -t[1]
        )
        placed = [None] * len(rects)          # None or (x,y,w,h,rotated)
        skyline = [(0, 0)]                    # list of (x, y) where a new rectangle can potentially start
        result = []

        def can_place(x, y, w, h):
            """Return True if a rectangle of size w,h can be placed at (x,y)."""
            if x + w > container_w or y + h > container_h:
                return False
            # Check overlap with existing placed rectangles
            for xx, yy, ww, hh, _ in placed:
                if xx is None:
                    continue
                if not (x + w <= xx or xx + ww <= x or y + h <= yy or yy + hh <= y):
                    return False
            return True

        for idx, area, rect in indexed:
            w0, h0 = rect.width, rect.height
            for (w, h, rot) in (
                (w0, h0, False),
                (h0, w0, True)  # rotated
            ) if rect.rotatable and (w0 != h0) else ((w0, h0, False),):
                # scan all potential start positions:
                placed_flag = False
                for sx, sy in skyline:
                    if can_place(sx, sy, w, h):
                        placed[idx] = (sx, sy, w, h, rot)
                        result.append(RectanglePlacement(idx, sx, sy, rot))
                        # Update skyline: add new potential start at (sx+w, sy) and (sx, sy+h)
                        skyline.append((sx + w, sy))
                        skyline.append((sx, sy + h))
                        placed_flag = True
                        break
                if placed_flag:
                    break  # stop at first valid orientation/position
        return result