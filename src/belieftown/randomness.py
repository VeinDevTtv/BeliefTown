"""Random draws keyed by semantic event, not by branch-dependent draw count."""
import hashlib
import json
import numpy as np


def keyed_rng(seed: int, *key: object) -> np.random.Generator:
    payload=json.dumps([seed,*key],sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
    number=int.from_bytes(hashlib.blake2b(payload,digest_size=16).digest(),"little")
    return np.random.default_rng(number)
