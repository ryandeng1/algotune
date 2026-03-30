# solver.py
import base64
from typing import Any, Dict

class Solver:
    """
    Encoder that base64‑encodes the supplied plaintext.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Encode the plaintext using the Base64 algorithm.

        Parameters
        ----------
        problem : dict[str, Any]
            Must contain a key ``'plaintext'`` whose value is a ``bytes`` object.

        Returns
        -------
        dict[str, bytes]
            A dictionary with the single key ``'encoded_data'``.
        """
        return {"encoded_data": base64.b64encode(problem["plaintext"])}