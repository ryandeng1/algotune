from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]
        
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        actual_ciphertext = ciphertext[:-16]
        tag = ciphertext[-16:]
        
        return {"ciphertext": actual_ciphertext, "tag": tag}
