import base64
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        # Pull plaintext directly from the dictionary
        plaintext = problem["plaintext"]
        # Fast Base64 encoding
        encoded = base64.b64encode(plaintext)
        # Return result in the expected format
        return {"encoded_data": encoded}