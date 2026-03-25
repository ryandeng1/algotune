from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem["associated_data"]
        
        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(nonce, plaintext, aad)
        
        return {
            "ciphertext": ciphertext[:-16],
            "tag": ciphertext[-16:]
        }
