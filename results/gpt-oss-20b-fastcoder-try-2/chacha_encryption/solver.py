from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16


class Solver:
    def solve(self, problem: dict[str, bytes]) -> dict[str, bytes]:
        """
        Encrypt the plaintext using ChaCha20-Poly1305.

        :param problem: A dict with keys:
            - 'key': 32‑byte key
            - 'nonce': 12‑byte nonce
            - 'plaintext': bytes to encrypt
            - 'associated_data': optional authenticated data
        :return: {'ciphertext': bytes, 'tag': bytes}
        """
        # Extract inputs
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Create cipher object once
        chacha = ChaCha20Poly1305(key)

        # Perform encryption; the result is ciphertext + 16‑byte tag
        ct_tag = chacha.encrypt(nonce, plaintext, associated_data)

        # Split ciphertext and tag
        ciphertext = ct_tag[:-POLY1305_TAG_SIZE]
        tag = ct_tag[-POLY1305_TAG_SIZE:]

        return {"ciphertext": ciphertext, "tag": tag}