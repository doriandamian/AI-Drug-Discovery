import hashlib
import os

_CHUNK = 1 << 20  # 1 MiB


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(model_path: str) -> str:
    return model_path + ".sha256"


def write_hash(model_path: str) -> None:
    digest = _hash_file(model_path)
    with open(_sidecar(model_path), "w") as f:
        f.write(digest + "\n")


def verify_hash(model_path: str) -> None:
    sc = _sidecar(model_path)
    if not os.path.exists(sc):
        raise RuntimeError(
            f"Integrity sidecar not found: {sc}. "
            "Retrain the model to generate a trusted hash."
        )
    with open(sc) as f:
        expected = f.read().strip()
    actual = _hash_file(model_path)
    if actual != expected:
        raise RuntimeError(
            f"Model integrity check failed for {model_path}: "
            f"expected {expected}, got {actual}. "
            "The file may have been tampered with, retrain to recover."
        )
