# solver.py
import hashlib
from typing import Any, Dict

class Solver:
    """
    Computes the SHA-256 digest of a given plaintext byte string.

    The implementation uses Python's built‑in hashlib module which is
    highly optimised (C code).  The original implementation used the
    cryptography package which involved additional Python context
    management and is noticeably slower for this simple task.

    The `solve` method is deliberately lightweight and contains no
    unnecessary exception handling or control flow.
    """
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        # Extract the plaintext from the problem dictionary.
        plaintext = problem["plaintext"]

        # Compute the SHA‑256 digest using the fast hashlib wrapper.
        digest = hashlib.sha256(plaintext).digest()

        # Return the result in the expected JSON‑serialisable format.
        return {"digest": digest}