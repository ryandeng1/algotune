import base64
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        encoded_data = base64.b64encode(plaintext)
        return {"encoded_data": encoded_data}