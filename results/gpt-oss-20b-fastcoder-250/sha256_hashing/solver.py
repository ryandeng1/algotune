# solver.py
import hashlib
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Compute the SHA-256 hash of the provided plaintext using Python's
        built‑in hashlib module, which is highly optimized in C and
        considerably faster than the cryptography library for this simple
        operation.

        Parameters
        ----------
        problem : dict
            Should contain a single key 'plaintext' with a bytes object.

        Returns
        -------
        dict
            Dictionary with a single key 'digest' containing the 32‑byte
            SHA‑256 hash of the plaintext.
        """
        plaintext = problem["plaintext"]
        # hashlib.sha256 is implemented in C and is the fastest available
        digest = hashlib.sha256(plaintext).digest()
        return {"digest": digest}
