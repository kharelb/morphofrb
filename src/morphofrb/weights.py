from __future__ import annotations

import os
from pathlib import Path


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "morphofrb"
    return Path.home() / ".cache" / "morphofrb"


def _try_packaged_asset(filename: str) -> Path | None:
    try:
        from importlib.resources import files
        p = files("morphofrb").joinpath(f"assets/{filename}")
        if p.is_file():
            return Path(p)
    except Exception:
        pass
    return None


def get_weights_path(
    *,
    filename: str = "fine_tuned_chime_cat_2.pth",
    hf_repo: str = "bkharel/morphofrb-weight",
    revision: str = "main",
    force_download: bool = False,
) -> Path:
    """
    Returns a local filesystem path to model weights.

    Priority:
      1) packaged asset (if exists)
      2) cached download from Hugging Face
    """

    asset = _try_packaged_asset(filename)
    if asset is not None and not force_download:
        return asset

    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / f"{Path(filename).stem}-{revision}.pth"

    if local_path.exists() and local_path.stat().st_size > 0 and not force_download:
        return local_path

    try:
        from huggingface_hub import hf_hub_download  # optional dependency

        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)

    except Exception:
        import urllib.request

        url = f"https://huggingface.co/{hf_repo}/resolve/{revision}/{filename}"
        print(f"[morphofrb] Downloading model weights from:\n  {url}")
        urllib.request.urlretrieve(url, local_path)
        return local_path
