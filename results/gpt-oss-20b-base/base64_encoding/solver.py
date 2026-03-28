import base64

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, bytes]:
        """
        Encode the plaintext using Base64.

        Args:
            problem (dict): A dictionary containing a 'plaintext' key whose value is
                            a bytes-like object.

        Returns:
            dict: {'encoded_data': <Base64 encoded bytes>}
        """
        plaintext = problem.get('plaintext')
        # Assumes `plaintext` is already bytes; otherwise cast if necessary
        if plaintext is None:
            raise ValueError("Missing 'plaintext' key in problem")
        return {'encoded_data': base64.b64encode(plaintext)}