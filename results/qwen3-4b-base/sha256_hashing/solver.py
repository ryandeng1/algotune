import hashlib

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        digest = hashlib.sha256(plaintext).digest()
        return {"digest": digest}
