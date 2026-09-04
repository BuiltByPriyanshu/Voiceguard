"""Waveform augmentation for hardening bonafide training data against
real-world recording conditions (background noise, room reverb) that
ASVspoof's studio-clean bonafide clips never expose the model to.

Why this exists: the model classifies "genuine = clean like training data"
if bonafide examples are all studio-quality. A live demo mic in a normal
room -- let alone a noisy one -- looks unlike anything the model has seen
as bonafide, so it gets flagged as suspicious (see HANDOFF.md and
extract_clip_embeddings.py's docstring for the same problem observed with
teammate_ref.wav before calibration). Augmenting bonafide clips with noise
and reverb teaches "these acoustic conditions are still a genuine voice"
without needing to physically record in every possible room/mic.
"""
import numpy as np


def add_noise(waveform: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise at a target signal-to-noise ratio (dB)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(waveform ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(waveform.dtype)
    return waveform + noise


def add_reverb(waveform: np.ndarray, sr: int, room_scale: float = 1.0) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying impulse response --
    a cheap stand-in for room reverb, good enough to teach the model
    "reflections/reverb don't mean synthetic", not meant to be acoustically
    precise."""
    rir_len = int(0.15 * room_scale * sr)
    t = np.arange(rir_len)
    decay = np.exp(-t / (0.03 * room_scale * sr))
    rng = np.random.default_rng(42)
    rir = rng.normal(0, 1, rir_len).astype(np.float32) * decay
    rir[0] = 1.0  # keep the direct path dominant
    rir /= np.sum(np.abs(rir)) + 1e-8
    wet = np.convolve(waveform, rir, mode="full")[: len(waveform)]
    return wet.astype(waveform.dtype)


def normalize_peak(waveform: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(waveform)) + 1e-8
    return (waveform / m * peak).astype(waveform.dtype)


def augment_variants(waveform: np.ndarray, sr: int) -> list:
    """Return [(variant_name, augmented_waveform), ...] -- a handful of
    realistic real-world conditions derived from one clean recording."""
    variants = []
    variants.append(("noise_light", normalize_peak(add_noise(waveform, snr_db=25, seed=1))))
    variants.append(("noise_heavy", normalize_peak(add_noise(waveform, snr_db=12, seed=2))))
    variants.append(("reverb_small_room", normalize_peak(add_reverb(waveform, sr, room_scale=0.7))))
    variants.append(("reverb_large_room", normalize_peak(add_reverb(waveform, sr, room_scale=1.8))))
    noisy_reverb = add_reverb(add_noise(waveform, snr_db=18, seed=3), sr, room_scale=1.2)
    variants.append(("noise_and_reverb", normalize_peak(noisy_reverb)))
    return variants
