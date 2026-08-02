"""
Source time-course generation and three-component noise model.
"""

from .source_signals import generate_signal_batch
from .noise import (
    generate_pink_noise,
    generate_white_noise,
    generate_physiological_background,
    mix_noise_components,
    sample_snr_db,
)

__all__ = [
    'generate_signal_batch',
    'generate_pink_noise',
    'generate_white_noise',
    'generate_physiological_background',
    'mix_noise_components',
    'sample_snr_db',
]
