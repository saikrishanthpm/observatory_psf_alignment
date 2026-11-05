"""JAX backend extension for Optiland."""

from optiland_jax.register import list_available_backends, register_jax_backend

__all__ = ["register", "register_jax_backend", "list_available_backends"]


def register(force: bool = False) -> None:
    """Register ``'jax'`` with ``optiland.backend``."""
    register_jax_backend(force=force)
