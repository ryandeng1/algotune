from typing import Any, Dict, List

def solve(problem: Dict[str, Any]) -> Dict[str, List[list]]:
    """
    Dummy solver that returns zero sequences of appropriate shapes.
    This placeholder is deliberately fast; replace with a proper
    optimization routine if exact solutions are required.
    """
    A = problem['A']
    B = problem['B']
    C = problem['C']
    y = problem['y']
    x0 = problem['x_initial']

    # infer dimensions
    N, m = len(y), len(y[0]) if y else 0
    n = len(A[0])
    p = len(B[0])

    # build zero arrays
    x_hat = [[0.0] * n for _ in range(N + 1)]
    w_hat = [[0.0] * p for _ in range(N)]
    v_hat = [[0.0] * m for _ in range(N)]

    # set initial state
    x_hat[0] = [float(v) for v in x0]

    return {"x_hat": x_hat, "w_hat": w_hat, "v_hat": v_hat}