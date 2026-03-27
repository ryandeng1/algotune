import hashlib
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the builtin hashlib library,
        which is faster and has no external dependencies.
        """
        plaintext = problem["plaintext"]
        # hashlib expects bytes; if not, encode it (assuming utf-8).
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        digest = hashlib.sha256(plaintext).digest()
        return {"digest": digest}