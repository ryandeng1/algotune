import base64
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encode the plaintext using Base64.

        Args:
            problem (dict): Dictionary containing the key 'plaintext' with bytes data.

        Returns:
            dict: {'encoded_data': <base64-encoded bytes>}
        """
        plaintext = problem["plaintext"]
        # If plaintext is a str, convert to bytes
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        # Perform Base64 encoding
        encoded_data = base64.b64encode(plaintext)
        return {"encoded_data": encoded_data}