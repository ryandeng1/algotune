import typing
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem.get("associated_data", b'')
        
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]
        
        return {"ciphertext": ciphertext, "tag": tag}
