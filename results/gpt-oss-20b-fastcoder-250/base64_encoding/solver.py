# solver.py
import base64
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encode the given plaintext using standard Base64 encoding.
        The implementation is a thin wrapper around base64.b64encode,
        which is highly optimized in CPython, ensuring both correctness
        and speed.

        Parameters
        ----------
        problem : dict
            Expected to contain a key 'plaintext' with a bytes object.

        Returns
        -------
        dict
            A dictionary containing the key 'encoded_data' with the Base64
            encoded bytes.
        """
        plaintext = problem.get("plaintext", b"")
        # Directly use the C-optimized base64 encoder
        encoded_data = base64.b64encode(plaintext)
        return {"encoded_data": encoded_data}
