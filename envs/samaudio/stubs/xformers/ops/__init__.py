class AttentionBias:  # only used in isinstance checks / type hints
    pass


class _Unavailable:
    def __getattr__(self, name):
        raise RuntimeError(f"xformers.ops.fmha.{name} is unavailable on this platform (stub)")


fmha = _Unavailable()
