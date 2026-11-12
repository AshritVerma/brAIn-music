"""`muse2 sonify` — turn a recording's raw EEG into sound files.

Two ways of hearing the brain signal itself, all written as 16-bit /
44.1 kHz WAVs (the spec for stems handed to collaborators):

1. Audification: the cleaned EEG played back faster, so its rhythms move
   into the audible range. At x172 one EEG sample = one audio sample
   (256 Hz * 172 ~= 44.1 kHz): 10 Hz alpha becomes ~1.7 kHz. Slower speeds
   keep more duration but sit lower (x10: alpha ~100 Hz, beta ~200 Hz).
   Stereo: left = left-hemisphere sensors (TP9, AF7), right = TP10, AF8.

2. Wavetable: 1-second slices of EEG resampled into 2048-sample single
   cycles, stacked into a Serum/Vital-format wavetable (loads directly in
   Vital, which is free, or Serum). Each frame is made loop-clean so it plays
   as a pitched tone; scanning frames = moving through the recording.
   Plus a rendered drone (wavetable_drone.wav) so you have audio right away.

Cleaning (all FFT-domain, offline): drop < 1 Hz drift and > 45 Hz, which
removes 60 Hz mains hum, its harmonics, and DC; railed samples (|x| >= 999 uV,
the Muse's "no contact" value) are bridged by interpolation.
"""

from __future__ import annotations

import csv
import json
import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np

OUT_RATE = 44_100
EEG_RATE = 256
FRAME = 2048
CHANNELS = ("TP9", "AF7", "AF8", "TP10")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_eeg(folder: Path) -> tuple[np.ndarray, int]:
    path = folder / "eeg.csv"
    rows = []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        idx = [header.index(c) for c in CHANNELS]
        for row in r:
            try:
                rows.append([float(row[i]) for i in idx])
            except (ValueError, IndexError):
                continue
    if not rows:
        raise SystemExit(f"{path} has no EEG samples")
    rate = EEG_RATE
    meta = folder / "session.json"
    if meta.exists():
        try:
            rate = int(json.loads(meta.read_text())["eeg"]["sampling_rate"]) or EEG_RATE
        except Exception:
            pass
    return np.asarray(rows, dtype=np.float64).T, rate  # (4, n)


def write_wav16(path: Path, data: np.ndarray, rate: int = OUT_RATE, extra_chunk: Optional[bytes] = None) -> None:
    """data: (n,) mono or (n, ch) in [-1, 1]."""
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
    if not extra_chunk:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(pcm.shape[1])
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        return
    # Hand-built RIFF so the extra chunk (Serum's 'clm ' marker) sits between
    # 'fmt ' and 'data', where wavetable synths look for it.
    ch = pcm.shape[1]
    fmt = struct.pack("<HHIIHH", 1, ch, rate, rate * ch * 2, ch * 2, 16)
    body = pcm.tobytes()
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra_chunk + b"data" + struct.pack("<I", len(body)) + body
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


# ---------------------------------------------------------------------------
# DSP helpers (numpy only)
# ---------------------------------------------------------------------------


def bridge_railed(x: np.ndarray, rail: float = 999.0) -> np.ndarray:
    x = x.copy()
    bad = np.abs(x) >= rail
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def bandpass_fft(x: np.ndarray, rate: int, lo: float = 1.0, hi: float = 45.0) -> np.ndarray:
    n = len(x)
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, 1.0 / rate)
    keep = (f >= lo) & (f <= hi)
    # short cosine tapers at the edges so the band edges don't ring
    taper = np.ones_like(f)
    edge = 0.5
    lo_t = (f >= lo - edge) & (f < lo)
    hi_t = (f > hi) & (f <= hi + edge)
    taper[lo_t] = 0.5 - 0.5 * np.cos(np.pi * (f[lo_t] - (lo - edge)) / edge)
    taper[hi_t] = 0.5 + 0.5 * np.cos(np.pi * (f[hi_t] - hi) / edge)
    mask = np.where(keep, 1.0, 0.0) + np.where(lo_t | hi_t, taper, 0.0)
    return np.fft.irfft(spec * mask, n)


def resample_fft(x: np.ndarray, n_out: int) -> np.ndarray:
    """Band-limited resample of a 1-D signal to exactly n_out samples."""
    n = len(x)
    if n_out == n:
        return x.copy()
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=complex)
    k = min(len(spec), len(out))
    out[:k] = spec[:k]
    return np.fft.irfft(out, n_out) * (n_out / n)


def normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return x / m * (10 ** (peak_db / 20))


def soft_limit(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(drive * x) / np.tanh(drive)


def fades(x: np.ndarray, ms: float = 20.0, rate: int = OUT_RATE) -> np.ndarray:
    n = min(int(rate * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n)
    x = x.copy()
    x[:n] *= ramp[:, None] if x.ndim == 2 else ramp
    x[-n:] *= ramp[::-1][:, None] if x.ndim == 2 else ramp[::-1]
    return x


def channel_quality(x: np.ndarray, rate: int) -> float:
    """Lower is better: share of 1-100 Hz power that is mains hum, plus railing."""
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    tot = spec[(f >= 1) & (f < 100)].sum() or 1.0
    hum = spec[(f >= 55) & (f <= 65)].sum() / tot
    railed = float(np.mean(np.abs(x) >= 999))
    return hum + 2 * railed


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------


def audify(clean: np.ndarray, rate: int, speed: float) -> np.ndarray:
    """clean: (4, n) cleaned EEG. Returns (m, 2) stereo in [-1, 1]."""
    left = clean[0] + clean[1]   # TP9 + AF7
    right = clean[3] + clean[2]  # TP10 + AF8
    n_out = max(2, int(round(clean.shape[1] * OUT_RATE / (rate * speed))))
    st = np.stack([resample_fft(left, n_out), resample_fft(right, n_out)], axis=1)
    st = soft_limit(st / (np.percentile(np.abs(st), 99.5) or 1.0), drive=1.5)
    return fades(normalize(st), ms=15)


def wavetable(signal: np.ndarray, rate: int, frames: int = 64, slice_s: float = 1.0) -> np.ndarray:
    """Return (frames, FRAME) loop-clean single cycles from a cleaned 1-D EEG signal."""
    span = int(slice_s * rate)
    n = len(signal)
    if n < span:
        signal = np.tile(signal, int(np.ceil(span / max(n, 1))) + 1)[: span + 1]
        n = len(signal)
    starts = np.linspace(0, n - span, num=frames).astype(int)
    out = np.zeros((frames, FRAME))
    for i, s in enumerate(starts):
        seg = signal[s : s + span].astype(np.float64)
        seg = seg - np.linspace(seg[0], seg[-1], len(seg))  # end == start -> clean loop
        cyc = resample_fft(seg, FRAME)
        cyc -= cyc.mean()
        out[i] = normalize(cyc, peak_db=-0.5)
    return out


def serum_clm_chunk(frame: int = FRAME) -> bytes:
    payload = f"<!>{frame} 01000000 wavetable (muse2-music-lab)".encode("ascii")
    if len(payload) % 2:
        payload += b"\0"
    return b"clm " + struct.pack("<I", len(payload)) + payload


def render_drone(table: np.ndarray, seconds: float = 16.0, root_hz: float = 55.0) -> np.ndarray:
    """Scan the wavetable start->end over `seconds`; root + octave + fifth, detuned stereo."""
    frames, size = table.shape
    n = int(seconds * OUT_RATE)
    t = np.arange(n) / OUT_RATE
    pos = np.linspace(0, frames - 1, n)          # frame position over time
    f0 = np.floor(pos).astype(int)
    f1 = np.minimum(f0 + 1, frames - 1)
    mix = pos - f0
    out = np.zeros((n, 2))
    voices = [(root_hz, 1.0), (root_hz * 2, 0.6), (root_hz * 3, 0.35)]
    for hz, amp in voices:
        for ch, detune in enumerate((0.997, 1.003)):
            phase = (t * hz * detune) % 1.0
            idx = phase * size
            i0 = idx.astype(int) % size
            i1 = (i0 + 1) % size
            fr = idx - np.floor(idx)
            a = table[f0, i0] * (1 - fr) + table[f0, i1] * fr
            b = table[f1, i0] * (1 - fr) + table[f1, i1] * fr
            out[:, ch] += amp * (a * (1 - mix) + b * mix)
    env = np.minimum(1.0, np.minimum(t / 2.0, (seconds - t) / 3.0))  # 2 s in, 3 s out
    out *= env[:, None]
    return normalize(soft_limit(out / (np.max(np.abs(out)) or 1.0), 1.2), peak_db=-1.0)


def run(folder: Path, speeds: tuple[float, ...] = (10.0, 40.0, 172.0), frames: int = 64,
        out_dir: Optional[Path] = None) -> int:
    folder = Path(folder).expanduser().resolve()
    eeg, rate = load_eeg(folder)
    out = Path(out_dir) if out_dir else folder / "sonify"
    out.mkdir(parents=True, exist_ok=True)
    dur = eeg.shape[1] / rate
    print(f"[sonify] {folder.name}: {eeg.shape[1]} samples ({dur:.1f} s @ {rate} Hz)")

    clean = np.stack([bandpass_fft(bridge_railed(ch), rate) for ch in eeg])
    q = [channel_quality(ch, rate) for ch in eeg]
    for name, score in zip(CHANNELS, q):
        print(f"[sonify]   {name}: contact score {score:.2f} (lower is better)")
    if min(q) > 0.8:
        print("[sonify]   WARNING: every channel is mostly hum/railed -> results will mostly be noise floor."
              " Re-record with the headset seated.")

    made = []
    for s in speeds:
        st = audify(clean, rate, s)
        p = out / f"audify_x{int(s)}.wav"
        write_wav16(p, st)
        made.append((p, len(st) / OUT_RATE))

    best = int(np.argmin(q))
    table = wavetable(clean[best], rate, frames=frames)
    p_wt = out / f"wavetable_{CHANNELS[best]}_{frames}x{FRAME}.wav"
    write_wav16(p_wt, table.reshape(-1), extra_chunk=serum_clm_chunk())
    made.append((p_wt, None))

    drone = render_drone(table)
    p_dr = out / "wavetable_drone.wav"
    write_wav16(p_dr, drone)
    made.append((p_dr, len(drone) / OUT_RATE))

    (out / "README.txt").write_text(
        f"Sonified from {folder.name} ({dur:.1f} s of EEG, 16-bit / 44.1 kHz).\n\n"
        + "audify_xN.wav  - raw EEG sped up N times (L = TP9+AF7, R = TP10+AF8), 1-45 Hz band.\n"
        + f"{p_wt.name} - {frames} frames x {FRAME} samples, from {CHANNELS[best]} (cleanest channel).\n"
        + "                 Drag into Vital (free) or Serum's wavetable editor.\n"
        + "wavetable_drone.wav - that wavetable played as A1+A2+E3, scanning all frames over 16 s.\n"
    )
    for p, secs in made:
        print(f"[sonify] wrote {p.name}" + (f"  ({secs:.2f} s)" if secs else ""))
    print(f"[sonify] folder: {out}")
    return 0
