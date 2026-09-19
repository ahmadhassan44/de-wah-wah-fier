"""Placeholder for xformers, which has no macOS build.

SAM Audio's dependency perception_models imports xformers at module level but
the audio separation path never calls it (it uses PyTorch attention). Anything
that does reach for it fails loudly here instead of silently.
"""
