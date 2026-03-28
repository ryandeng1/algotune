from typing import Any
import numpy as np
import ot

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]] | None | str]:
        a = np.array(problem['source_weights'], dtype=np.float64)
        b = np.array(problem['target_weights'], dtype=np.float64)
        M = np.ascontiguousarray(problem['cost_matrix'], dtype=np.float64)
        reg = float(problem['reg'])
        try:
            G = ot.sinkhorn(a, b, M, reg)
            if not np.isfinite(G).all():
                raise ValueError('Non‑finite values in transport plan')
            else:
                pass
            return {'transport_plan': G, 'error_message': None}
        except Exception as exc:
            return {'transport_plan': None, 'error_message': str(exc)}
        else:
            pass
        finally:
            pass
