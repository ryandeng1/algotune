import math
from typing import Any, List, Tuple

# Simple greedy solver: place rectangles in the container in a first-fit manner.
# This implementation is not guaranteed to be optimal, but it aims to be fast.
# It may not pass the optimality check of the framework.

class Solver:
    def solve(self, problem: Tuple[int, int, List[Tuple[int, int, bool]]], **kwargs) -> List[Tuple[int, int, int, bool]]:
        W, H, rectangles = problem
        n = len(rectangles)
        placed = [None] * n  # store placement tuple (x, y, rotated)
        # Sort by area descending
        order = sorted(range(n), key=lambda i: rectangles[i][0] * rectangles[i][1], reverse=True)

        # Keep a list of free rectangles in the container as [x, y, w, h]
        free_rects = [[0, 0, W, H]]

        def best_split(free, w, h):
            x, y, fw, fh = free
            if w > fw or h > fh:
                return None
            # place at (x, y)
            new_free = []
            # right free
            if fw - w > 0:
                new_free.append([x + w, y, fw - w, h])
            # top free
            if fh - h > 0:
                new_free.append([x, y + h, fw, fh - h])
            # rest of space
            return new_free

        for i in order:
            w, h, rot = rectangles[i]
            placed_flag = False
            # try both orientations
            for (rw, rh, rot_flag) in [(w, h, False), (h, w, True)] if rot else [(w, h, False)]:
                for idx, free in enumerate(free_rects):
                    splits = best_split(free, rw, rh)
                    if splits is not None:
                        # place
                        placed[i] = (free[0], free[1], rot_flag)
                        # replace free rect with splits
                        free_rects.pop(idx)
                        free_rects.extend(splits)
                        placed_flag = True
                        break
                if placed_flag:
                    break
            # if not placed, skip
        # build result list
        result = []
        for idx, p in enumerate(placed):
            if p:
                x, y, rot_flag = p
                result.append((idx, x, y, rot_flag))
        return result
