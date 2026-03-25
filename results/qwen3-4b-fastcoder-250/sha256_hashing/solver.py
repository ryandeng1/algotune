from cryptography.hazmat.primitives import hashes
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        try:
            return {"digest": hashes.Hash(hashes.SHA256()).update(plaintext).finalize()}
        except Exception as e:
            raise
