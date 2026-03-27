from typing import Any, NamedTuple, List
import itertools

class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool

class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: List[Rectangle]

class RectanglePlacement(NamedTuple):
    i: int  # Index of rectangle
    x: int  # Bottom-left x coordinate
    y: int  # Bottom-left y coordinate
    rotated: bool  # Whether the rectangle is rotated

def solve(problem: Instance) -> List[RectanglePlacement]:
    """
    Very small heuristic: place rectangles greedily in rows starting at the
    bottom-left corner.  If a rectangle does not fit in the remaining width
    of the current row it is placed in the next row.  Rotations are chosen
    only when the rectangle is rotatable and fits better in the current
    orientation.
    """
    # Sort by height descending to reduce fragmentation
    rects = list(enumerate(problem.rectangles))
    rects.sort(key=lambda p: max(p[1].width, p[1].height), reverse=True)

    placements: List[RectanglePlacement] = []

    x, y = 0, 0
    row_max_height = 0
    for idx, rect in rects:
        # choose orientation
        w, h = rect.width, rect.height
        if rect.rotatable and w > h and w <= problem.container_width - x:
            w, h = h, w

        if w > problem.container_width - x:
            # start new row
            y += row_max_height
            if y + h > problem.container_height:
                break  # no more space
            x = 0
            row_max_height = 0

        placements.append(RectanglePlacement(idx, x, y, w != rect.width))
        x += w
        row_max_height = max(row_max_height, h)

    return placements