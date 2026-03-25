import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        # Extract and convert the matrix
        raw = problem.get("matrix")
        if raw is None:
            return {"sqrtm": {"X": []}}
        # Convert stringified complex numbers or native complex to ndarray
        try:
            # if elements are already complex or convertible numbers
            A = np.array(raw, dtype=complex)
        except Exception:
            # fallback: parse each element string manually
            parsed = [
                [complex(str(elem)) if isinstance(elem, str) else complex(elem)
                 for elem in row]
                for row in raw
            ]
            A = np.array(parsed, dtype=complex)

        # compute the principal square root
        try:
            X, _ = scipy.linalg.sqrtm(A, disp=False)
        except Exception:
            # propagate failure as empty list
            return {"sqrtm": {"X": []}}

        # serialize to list of lists of complex numbers
        return {"sqrtm": {"X": X.tolist()}}
