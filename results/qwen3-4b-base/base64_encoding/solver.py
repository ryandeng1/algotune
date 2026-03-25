import base64
from typing import Dict

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        return {"encoded_data": base64.b64encode(problem["plaintext"])}
