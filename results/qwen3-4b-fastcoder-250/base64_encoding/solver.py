import base64

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        return {"encoded_data": base64.b64encode(plaintext)}
