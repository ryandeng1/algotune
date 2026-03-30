from typing import Any, Dict
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Set of allowed key sizes for quick membership test
_VALID_KEY_SIZES = {16, 24, 32}
_TAG_SIZE = 16

class Solver:
    """Fast AES‑GCM encryption wrapper using a per‑key cache for AESGCM objects."""
    def __init__(self) -> None:
        # Cache to avoid re‑creating AESGCM objects for the same key.
        # The cache size is limited implicitly by GC; in most workloads
        # this is a significant win because key creation is expensive.
        self._aes_cache: Dict[bytes, AESGCM] = {}

    def _get_aesgcm(self, key: bytes) -> AESGCM:
        """Return a cached AESGCM instance for *key*, creating it if necessary."""
        aes = self._aes_cache.get(key)
        if aes is None:
            aes = AESGCM(key)
            self._aes_cache[key] = aes
        return aes

    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Encrypt the plaintext using AES‑GCM.

        Args:
            problem: dict with keys 'key', 'nonce', 'plaintext', 'associated_data'

        Returns:
            dict with 'ciphertext' (bytes) and 'tag' (bytes)
        """
        key = problem['key']
        # Basic key size validation; optional but cheap
        if len(key) not in _VALID_KEY_SIZES:
            raise ValueError(f'Invalid key size: {len(key)}. Must be one of {_VALID_KEY_SIZES}.')

        # Acquire the AESGCM instance (cached per key)
        aesgcm = self._get_aesgcm(key)

        # Perform encryption in a single call
        ciphertext_with_tag = aesgcm.encrypt(
            problem['nonce'],
            problem['plaintext'],
            problem['associated_data']
        )

        # Split ciphertext and tag
        # AESGCM returns ciphertext + tag; tag length is 16 bytes
        tag = ciphertext_with_tag[-_TAG_SIZE:]
        ciphertext = ciphertext_with_tag[:-_TAG_SIZE]

        return {'ciphertext': ciphertext, 'tag': tag}