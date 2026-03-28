from typing import Any, Dict
import hashlib

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        # Compute SHA-256 digest using the lightweight hashlib implementation
        plaintext = problem["plaintext"]
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        digest = hashlib.sha256(plaintext).digest()
        return {"digest": digest}