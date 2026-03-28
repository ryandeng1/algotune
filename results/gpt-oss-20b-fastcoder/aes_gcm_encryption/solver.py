import cryptography
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    GCM_TAG_SIZE = 16  # bytes

    def solve(self, problem):
        """
        Encrypt the plaintext using AES-GCM (cryptography library).

        Parameters
        ----------
        problem : dict
            Must contain keys 'key', 'nonce', 'plaintext', and 'associated_data'
            (each a bytes-like object of appropriate length).

        Returns
        -------
        dict
            {'ciphertext': <bytes>, 'tag': <bytes>}
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # AESGCM construction is cheap; no extra validation needed
        aesgcm = AESGCM(key)
        # encrypt() returns ciphertext concatenated with the tag
        ct_and_tag = aesgcm.encrypt(nonce, plaintext, associated_data)

        # Split into ciphertext and tag
        return {
            "ciphertext": ct_and_tag[:-self.GCM_TAG_SIZE],
            "tag": ct_and_tag[-self.GCM_TAG_SIZE:],
        }