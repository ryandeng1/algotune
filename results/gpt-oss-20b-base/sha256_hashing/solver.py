# solver.py

import hashlib
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the standard library's
        hashlib implementation, which is faster than the cryptography wrapper
        used in the reference solution.

        Parameters
        ----------
        problem : dict
            Dictionary containing the key "plaintext" with a bytes object.

        Returns
        -------
        dict
            Dictionary with the key "digest" containing the 32‑byte SHA‑256 hash.
        """
        plaintext = problem["plaintext"]
        digest = hashlib.sha256(plaintext).digest()
        return {"digest": digest}
