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
    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        # Simple first‑fit greedy placement: place each rectangle at the next available
        # top‑left position, rotating if allowed and fits better.
        placements: List[RectanglePlacement] = []
        used = []
        for idx, rect in enumerate(problem.rectangles):
            best_pos = None
            best_rot = False
            for y in range(problem.container_height - min(rect.height, rect.width) + 1):
                for x in range(problem.container_width - min(rect.width, rect.height) + 1):
                    # Test rotation
                    for rot in (False, True) if rect.rotatable else (False,):
                        w, h = (rect.width, rect.height) if not rot else (rect.height, rect.width)
                        if x + w > problem.container_width or y + h > problem.container_height:
                            continue
                        # Check collision
                        collision = False
                        for ux, uy, w2, h2, _ in used:
                            if not (x + w <= ux or ux + w2 <= x or y + h <= uy or uy + h2 <= y):
                                collision = True
                                break
                        if not collision:
                            best_pos = (x, y)
                            best_rot = rot
                            break
                if best_pos is not None:
                    break
            if best_pos:
                x, y = best_pos
                w, h = (rect.width, rect.height) if not best_rot else (rect.height, rect.width)
                placements.append(RectanglePlacement(idx, x, y, best_rot))
                used.append((x, y, w, h, rect.rotatable))
        return placements

    def _typesafe_instance(self, instance):
        if isinstance(instance, Instance):
            return instance
        return Instance(instance[0], instance[1], [Rectangle(*r) for r in instance[2]])