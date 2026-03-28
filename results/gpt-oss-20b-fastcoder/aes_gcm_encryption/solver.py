from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    def solve(self, problem: dict) -> dict:
        key, nonce, plaintext, aad = problem['key'], problem['nonce'], problem['plaintext'], problem['associated_data']
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext, aad)
        return {'ciphertext': ct[:-16], 'tag': ct[-16:]}