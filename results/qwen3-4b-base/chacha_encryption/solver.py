from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]
        
        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
        
        actual_ciphertext = ciphertext[:-16]
        tag = ciphertext[-16:]
        
        return {"ciphertext": actual_ciphertext, "tag": tag}
