import hashlib
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        # Compute SHA-256 hash using the fast built‑in hashlib
        plaintext = problem["plaintext"]
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes or bytearray")
        return {"digest": hashlib.sha256(plaintext).digest()}