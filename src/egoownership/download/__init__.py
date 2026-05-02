"""Dataset-specific download helpers.

Each module exposes a ``download(out_dir: Path, **kwargs) -> None`` entrypoint
that either downloads annotations directly (for public files) or prints the
exact CLI commands the user must run themselves (for license-gated datasets).
"""
