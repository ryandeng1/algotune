from __future__ import annotations

import math
import typing as _t

Num = _t.Union[int, float]
Array = _t.Sequence[Num]
Problem = _t.Dict[str, _t.Union[Array, Num]]


class Result(_t.TypedDict):
    success: bool
    y: _t.List[Array]
    message: str | None


class Solver:
    """
    Very small ODE solver that integrates a linear system

          dy/dt = A(t) @ y + b(t)
    with a simple explicit Runge–Kutta 4th‑order method.
    No external libraries (besides the stdlib) are required – all
    array arithmetic is done with Python lists to avoid the cost of
    constructing temporary numpy arrays.
    """

    @staticmethod
    def _rk4_step(
        A: _t.Callable[[Num], _t.List[_t.List[Num]]],
        b: _t.Callable[[Num], _t.List[Num]],
        y: _t.List[Num],
        t: Num,
        h: Num,
    ) -> _t.List[Num]:
        """Perform a single RK4 step with dense matrix (list of lists)."""
        y = list(map(float, y))
        # k1 = f(t , y)
        k1 = Solver._matrix_vec(A(t), y, b(t))
        # k2 = f(t+h/2, y + h/2*k1)
        y_half = [y_i + h / 2.0 * k1_i for y_i, k1_i in zip(y, k1)]
        k2 = Solver._matrix_vec(A(t + h / 2.0), y_half, b(t + h / 2.0))
        # k3
        y_half = [y_i + h / 2.0 * k2_i for y_i, k2_i in zip(y, k2)]
        k3 = Solver._matrix_vec(A(t + h / 2.0), y_half, b(t + h / 2.0))
        # k4
        y_full = [y_i + h * k3_i for y_i, k3_i in zip(y, k3)]
        k4 = Solver._matrix_vec(A(t + h), y_full, b(t + h))

        # Combine
        return [
            y_i + h / 6.0 * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
            for y_i, k1_i, k2_i, k3_i, k4_i in zip(y, k1, k2, k3, k4)
        ]

    @staticmethod
    def _matrix_vec(
        M: _t.List[_t.List[Num]], v: _t.List[Num], alt: _t.List[Num]
    ) -> _t.List[Num]:
        """Compute M @ v + alt for dense matrices given as lists."""
        n = len(v)
        res = [0.0] * n
        for i in range(n):
            row = M[i]
            s = alt[i]
            for j in range(n):
                s += row[j] * v[j]
            res[i] = s
        return res

    @staticmethod
    def _time_grid(
        t0: Num, tf: Num, steps: int
    ) -> _t.List[Num]:
        h = (tf - t0) / steps
        return [t0 + i * h for i in range(steps + 1)], h

    def _solve(
        self,
        problem: Problem,
        debug: bool = False,
    ) -> Result:
        """
        Expected problem dict keys:
            * A: function(t) -> 2‑d list (nxn matrix)
            * b: function(t) -> 1‑d list (length n)
            * y0: list (length n)
            * t_span: (t0, tf)
            * t_steps: int (number of sub‑steps, default 1000)
        """
        A = problem["A"]
        b = problem["b"]
        y0 = list(map(float, problem["y0"]))
        t0, tf = problem["t_span"]
        steps = int(problem.get("t_steps", 1000))

        grid, h = self._time_grid(t0, tf, steps)
        y = y0
        for i in range(steps):
            y = self._rk4_step(A, b, y, grid[i], h)
        return Result(success=True, y=[y], message=None)

    def solve(
        self, problem: Problem
    ) -> _t.Dict[str, _t.List[Num]]:
        sol = self._solve(problem, debug=False)

        # Extract final state
        if sol["success"]:
            return {"final_state": sol["y"][0]}
        else:
            raise RuntimeError(f"Solver failed: {sol['message']}")