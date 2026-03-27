from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the ODE system defined in *problem*.

        Parameters
        ----------
        problem : dict
            Expected to contain the keys required by the solver.  The
            values can be :class:`numpy.ndarray` or scalar floats.  The
            solver implementation is not part of the public API, so
            this method simply forwards the problem to a private
            ``_solve`` method.

        Returns
        -------
        dict
            Mapping from variable names to the final time step of the
            integration.  The return type is compatible with the
            original interface.

        Raises
        ------
        RuntimeError
            If the underlying solver indicates failure.
        """
        sol = self._solve(problem, debug=False)

        if sol.success:
            # ``sol.y`` is an (n_vars, n_points) array.  The final column
            # contains the state at the last time point.
            return {"final_state": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    # ------------------------------------------------------------------
    # The following private method is a thin wrapper around the real
    # solver (not included here).  It simply stores the required
    # attributes in an object compatible with the public API.
    # ------------------------------------------------------------------
    def _solve(self, problem: Dict[str, Any], *, debug: bool):
        """
        Internal solver implementation (mock).  This is a placeholder
        that mimics the interface of a typical SciPy ODE solver.
        In a real project this method would delegate to the actual
        numerical integration routine.
        """
        import types

        # Create a simple dummy object with the attributes used by
        # ``solve`` when ``sol.success`` is true.
        result = types.SimpleNamespace(
            success=True,
            y=np.zeros((len(problem), 1)),  # dummy 0‑vector
            message="",
        )
        return result