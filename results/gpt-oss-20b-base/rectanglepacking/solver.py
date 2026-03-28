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
    """Fast heuristic solver: first‑fit with optional rotation."""

    def _typesafe_instance(self, instance):
        if isinstance(instance, Instance):
            return instance
        return Instance(instance[0], instance[1],
                         [Rectangle(*r) for r in instance[2]])

    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        problem = self._typesafe_instance(problem)
        w0, h0 = problem.container_width, problem.container_height
        rects = problem.rectangles

        # sort by area descending to pack bigger first
        order = sorted(range(len(rects)),
                       key=lambda i: rects[i].width * rects[i].height,
                       reverse=True)

        # current skyline: list of (x, y, width)
        skyline = [(0, 0, w0)]
        placed = []

        for idx in order:
            rect = rects[idx]
            best = None
            best_size = None

            # try both orientations
            for rot in (False, True) if rect.rotatable else (False,):
                rw, rh = (rect.height, rect.width) if rot else (rect.width, rect.height)
                for seg_idx, (sx, sy, sw) in enumerate(skyline):
                    if rw <= sw:
                        # place at sx, sy
                        new_y = sy
                        # ensure rectangle height fits in container
                        if new_y + rh > h0:
                            continue
                        size = rw * rh
                        if best is None or size > best_size:
                            best = (seg_idx, sx, new_y, rw, rh, rot)
                            best_size = size

            if best:
                seg_idx, cx, cy, rw, rh, rot = best
                # update skyline: split segment
                seg_x, seg_y, seg_w = skyline[seg_idx]
                before = skyline[:seg_idx]
                after = skyline[seg_idx + 1:]
                skyline = before
                if seg_w > rw:
                    skyline.append((seg_x + rw, seg_y, seg_w - rw))
                skyline += after
                # merge adjacent same y
                merged = []
                for x, y, w in skyline:
                    if merged and merged[-1][1] == y and merged[-1][2] == w:
                        merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + w)
                    else:
                        merged.append((x, y, w))
                skyline = merged
                placed.append(RectanglePlacement(idx, cx, cy, rot))
                # update vertical position of skyline segments that overlap placed rect
                for i, (x, y, w) in enumerate(skyline):
                    if x < cx + rw <= x + w:
                        skyline[i] = (x, cy + rh, w)
                # re‑sort skyline by x to keep order
                skyline.sort()

        return placed