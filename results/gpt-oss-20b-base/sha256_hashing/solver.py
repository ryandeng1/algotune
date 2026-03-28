import hashlib
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the stdlib hashlib.
        This implementation is faster and more lightweight than the cryptography
        based approach.
        """
        plaintext = problem["plaintext"]
        # hashlib expects bytes; if plaintext is a string encode it (utf-8 by default)
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        # Compute digest in one step (fast, no Python-level looping)
        hash_value = hashlib.sha256(plaintext).digest()
        return {"digest": hash_value}