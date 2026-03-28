from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    # Pre‑computed constants
    _TAG_LEN = 16
    _VALID_KEY_SIZES = {16, 24, 32}

    def solve(self, problem: dict[str, bytes]) -> dict[str, bytes]:
        """
        Encrypt the plaintext using AES-GCM from the `cryptography` library.

        Args:
            problem (dict): Dictionary containing 'key', 'nonce', 'plaintext',
                            and 'associated_data'.

        Returns:
            dict: Contains 'ciphertext' and 'tag'.
        """
        key, nonce, plaintext, ad = (
            problem['key'],
            problem['nonce'],
            problem['plaintext'],
            problem['associated_data'],
        )

        if len(key) not in self._VALID_KEY_SIZES:
            raise ValueError(
                f"Invalid key size: {len(key)}. Must be one of {sorted(self._VALID_KEY_SIZES)}."
            )

        aesgcm = AESGCM(key)
        enc = aesgcm.encrypt(nonce, plaintext, ad)

        # Separate plaintext and tag
        cipher = enc[:-self._TAG_LEN]
        tag = enc[-self._TAG_LEN:]

        return {'ciphertext': cipher, 'tag': tag}