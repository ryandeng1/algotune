import hashlib
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using hashlib for maximum speed.
        """
        plaintext = problem["plaintext"]
        # hashlib works directly on bytes, so no exception handling is needed
        hash_value = hashlib.sha256(plaintext).digest()
        return {"digest": hash_value}