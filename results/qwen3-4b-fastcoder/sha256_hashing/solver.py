from typing import Any
from cryptography.hazmat.primitives import hashes


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        return {"digest": hashes.Hash(hashes.SHA256()).update(problem["plaintext"]).finalize()}