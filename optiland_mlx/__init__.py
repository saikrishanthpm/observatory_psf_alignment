"""MLX backend extension for Optiland (Apple Silicon / spaceklip_new)."""

from optiland_mlx.register import list_available_backends, register_mlx_backend

__all__ = ["register", "register_mlx_backend", "list_available_backends"]


def register(force: bool = False) -> None:
    """Register ``'mlx'`` with ``optiland.backend``."""
    register_mlx_backend(force=force)
