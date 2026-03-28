from typing import Any
import hashlib

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the built‑in hashlib library.
        """
        plaintext = problem['plaintext']
        # hashlib expects bytes, so ensure we get a bytes object.
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes-like")
        # Compute the digest in a single call.
        hash_value = hashlib.sha256(plaintext).digest()
        return {'digest': hash_value}