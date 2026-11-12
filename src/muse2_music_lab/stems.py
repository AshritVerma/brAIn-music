"""Project-ready stems from EEG, plus a realistic EEG simulator.

`muse2 stems` renders three files sized and tuned for a Logic project:

  muse_drone_<key>.wav   EEG wavetable played as a drone on the project key
                         (root + octave + fifth), scanning the whole recording,
                         exactly `bars` long at the project tempo.
  muse_texture.wav       audification of the EEG (stereo hemispheres), sped up
                         so the whole recording fills the same `bars`.
  muse_hits_<bpm>bpm.wav eight percussive one-shots cut from the audified EEG at
                         blink / jaw-clench moments, one per beat, so Quick
                         Sampler's Slice mode or Flex can split them instantly.

`muse2 simulate-eeg` writes a recording folder of synthetic-but-plausible EEG
(1/f background, waxing/waning alpha, theta, beta, blinks, jaw-clench EMG),
clearly labelled SIMULATED, for building the piece before real data exists.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from muse2_music_lab.sonify import (
    CHANNELS,
    OUT_RATE,
    bandpass_fft,
    bridge_railed,
    channel_quality,
    fades,
    load_eeg,
    normalize,
    render_drone,
    resample_fft,
    soft_limit,
    wavetable,
    write_wav16,
)

NOTE_HZ = {  # octave 2 (C2 = 65.41 Hz)
    "C": 65.41, "C#": 69.30, "Db": 69.30, "D": 73.42, "D#": 77.78, "Eb": 77.78,
    "E": 82.41, "F": 87.31, "F#": 92.50, "Gb": 92.50, "G": 98.00, "G#": 103.83,
    "Ab": 103.83, "A": 110.00, "A#": 116.54, "Bb": 116.54, "B": 123.47,
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def simulate_eeg(seconds: float, rate: int = 256, seed: Optional[int] = None) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate

    def pink(scale: float) -> np.ndarray:
        spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
        f = np.fft.rfftfreq(n, 1 / rate)
        spec /= np.sqrt(np.maximum(f, 0.5))
        x = np.fft.irfft(spec, n)
        return x / x.std() * scale

    def slow_env(period_s: float) -> np.ndarray:
        k = max(4, int(seconds / period_s) + 2)
        knots = rng.uniform(0, 1, k)
        return np.interp(t, np.linspace(0, seconds, k), knots)

    alpha_env = 0.25 + 0.75 * slow_env(8.0) ** 1.2   # alpha waxes and wanes, never fully gone
    focus_env = 1 - alpha_env                    # beta loosely anti-correlated
    alpha_f = 10 + 0.4 * np.sin(2 * np.pi * t / 37)
    alpha = np.sin(2 * np.pi * np.cumsum(alpha_f) / rate)
    theta = np.sin(2 * np.pi * 6.2 * t + rng.uniform(0, 6))
    beta_f = 19 + 2.5 * slow_env(5.0)                # beta wanders 19-21.5 Hz
    beta = np.sin(2 * np.pi * np.cumsum(beta_f) / rate) + 0.5 * np.sin(2 * np.pi * np.cumsum(beta_f * 0.85) / rate)

    gains = {  # (alpha, theta, beta) per channel: alpha strongest at temporal-posterior sites
        "TP9": (22, 4, 3), "AF7": (9, 7, 5), "AF8": (9, 7, 5), "TP10": (22, 4, 3),
    }
    eeg = np.zeros((4, n))
    for i, ch in enumerate(CHANNELS):
        ga, gt, gb = gains[ch]
        eeg[i] = (
            pink(9.0)
            + ga * alpha_env * alpha
            + gt * slow_env(12.0) * theta
            + gb * (0.4 + focus_env) * beta
            + 25 * np.sin(2 * np.pi * t / rng.uniform(20, 60) + i)   # slow electrode drift
            + 3 * np.sin(2 * np.pi * 60 * t)                          # a little mains hum
        )

    events = []
    tt = rng.uniform(2, 5)
    while tt < seconds - 1:                      # blinks: big slow frontal deflections
        amp = rng.uniform(150, 320)
        w = np.exp(-0.5 * ((t - tt) / 0.07) ** 2) - 0.25 * np.exp(-0.5 * ((t - tt - 0.2) / 0.12) ** 2)
        eeg[1] += amp * w
        eeg[2] += amp * 1.05 * w
        eeg[0] += amp * 0.12 * w
        eeg[3] += amp * 0.12 * w
        events.append((round(tt, 3), "blink"))
        tt += rng.uniform(3, 8)
    tt = rng.uniform(6, 12)
    while tt < seconds - 1:                      # jaw clench: broadband EMG burst
        dur = rng.uniform(0.25, 0.6)
        m = (t >= tt) & (t < tt + dur)
        burst = rng.standard_normal(m.sum())
        burst = np.diff(np.concatenate([[0], burst]))   # tilt towards high frequencies
        env = np.hanning(m.sum())
        for i in range(4):
            eeg[i, m] += rng.uniform(60, 120) * burst * env * (1.4 if i in (0, 3) else 1.0)
        events.append((round(tt, 3), "jaw"))
        tt += rng.uniform(8, 15)
    events.sort()
    return np.clip(eeg, -999.0, 999.0), events


def write_simulated_recording(out_root: Path, seconds: float, seed: Optional[int] = None) -> Path:
    rate = 256
    eeg, events = simulate_eeg(seconds, rate, seed)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(out_root) / f"{stamp}_SIMULATED-eeg"
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(folder / "eeg.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unix_time", "package_num", *CHANNELS])
        for i in range(eeg.shape[1]):
            w.writerow([f"{t0 + i / rate:.6f}", (i // 12) % 65536, *(f"{v:.3f}" for v in eeg[:, i])])
    with open(folder / "markers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unix_time", "elapsed_s", "text"])
        for et, kind in events:
            w.writerow([f"{t0 + et:.6f}", f"{et:.3f}", f"simulated {kind}"])
    (folder / "session.json").write_text(json.dumps({
        "folder": str(folder),
        "label": "SIMULATED EEG",
        "simulated": True,
        "note": "Synthetic EEG model (1/f background, alpha/theta/beta, blinks, jaw EMG). NOT a brain recording.",
        "duration_s": seconds,
        "eeg": {"source": "SIMULATED", "sampling_rate": rate, "channels": list(CHANNELS),
                "units": "microvolts", "samples": int(eeg.shape[1])},
        "events": [{"t": et, "kind": k} for et, k in events],
    }, indent=2))
    print(f"[simulate] wrote {seconds:.0f} s of SIMULATED EEG ({len(events)} blink/jaw events) -> {folder}")
    return folder


# ---------------------------------------------------------------------------
# Event detection on real or simulated EEG
# ---------------------------------------------------------------------------


def detect_events(eeg: np.ndarray, rate: int) -> list[tuple[float, str]]:
    """Blinks: big low-frequency frontal peaks. Jaw: bursts of 20-45 Hz power on all channels."""
    events: list[tuple[float, str]] = []
    frontal = bandpass_fft(bridge_railed(eeg[1]) + bridge_railed(eeg[2]), rate, 0.5, 6.0)
    thr = np.median(np.abs(frontal)) * 6 + 1e-9
    i, n = 0, len(frontal)
    while i < n:
        if frontal[i] > thr:
            j = min(n, i + int(0.4 * rate))
            pk = i + int(np.argmax(frontal[i:j]))
            events.append((pk / rate, "blink"))
            i = j + int(0.3 * rate)
        else:
            i += 1
    hf = np.mean([np.abs(bandpass_fft(bridge_railed(ch), rate, 20.0, 45.0)) for ch in eeg], axis=0)
    k = int(0.1 * rate)
    env = np.convolve(hf, np.ones(k) / k, mode="same")
    thr = np.median(env) * 4 + 1e-9
    i = 0
    while i < n:
        if env[i] > thr:
            j = min(n, i + int(0.6 * rate))
            pk = i + int(np.argmax(env[i:j]))
            events.append((pk / rate, "jaw"))
            i = j + int(0.5 * rate)
        else:
            i += 1
    return sorted(events)


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------


def _hit(grain: np.ndarray, length: int) -> np.ndarray:
    g = grain[:length].copy()
    if len(g) < length:
        g = np.pad(g, ((0, length - len(g)), (0, 0)))
    tt = np.arange(length) / OUT_RATE
    env = np.minimum(1.0, tt / 0.003) * np.exp(-tt / 0.09)
    g = g * env[:, None]
    return normalize(soft_limit(g / (np.max(np.abs(g)) or 1.0), 1.6), peak_db=-1.0)


def render_stems(folder: Path, key: str = "C", bpm: float = 105.0, bars: int = 16,
                 beats_per_bar: int = 4, out_dir: Optional[Path] = None) -> list[Path]:
    folder = Path(folder).expanduser().resolve()
    eeg, rate = load_eeg(folder)
    sim = False
    meta = folder / "session.json"
    if meta.exists():
        try:
            sim = bool(json.loads(meta.read_text()).get("simulated"))
        except Exception:
            pass
    tag = "_SIMULATED" if sim else ""
    out = Path(out_dir) if out_dir else folder / "stems"
    out.mkdir(parents=True, exist_ok=True)
    seconds = bars * beats_per_bar * 60.0 / bpm
    n_target = int(round(seconds * OUT_RATE))
    root = NOTE_HZ.get(key[:1].upper() + key[1:], NOTE_HZ["C"])

    raw = np.stack([bridge_railed(ch) for ch in eeg])
    clean = np.stack([bandpass_fft(ch, rate) for ch in raw])
    q = [channel_quality(ch, rate) for ch in eeg]
    best = int(np.argmin(q))
    made = []

    # 1) drone on the project key, exactly `bars` long
    table = wavetable(clean[best], rate, frames=64)
    drone = render_drone(table, seconds=seconds, root_hz=root)
    p = out / f"muse_drone_{key}{tag}.wav"
    write_wav16(p, drone)
    made.append(p)

    # 2) texture: whole recording audified to fill `bars`
    left, right = clean[0] + clean[1], clean[3] + clean[2]
    st = np.stack([resample_fft(left, n_target), resample_fft(right, n_target)], axis=1)
    st = soft_limit(st / (np.percentile(np.abs(st), 99.5) or 1.0), 1.5)
    speed = eeg.shape[1] / rate / seconds
    p = out / f"muse_texture_x{speed:.0f}{tag}.wav"
    write_wav16(p, fades(normalize(st), ms=40))
    made.append(p)

    # 3) hits: grains of x40 audification (keeps EMG crunch, 1-120 Hz) at events, one per beat
    speed_h = 40.0
    wide = np.stack([bandpass_fft(ch, rate, 1.0, 120.0) for ch in raw])
    lh, rh = wide[0] + wide[1], wide[3] + wide[2]
    n_h = int(round(eeg.shape[1] * OUT_RATE / (rate * speed_h)))
    aud = np.stack([resample_fft(lh, n_h), resample_fft(rh, n_h)], axis=1)
    events = detect_events(eeg, rate)
    blinks = [e for e in events if e[1] == "blink"]
    jaws = [e for e in events if e[1] == "jaw"]
    order: list[tuple[float, str]] = []
    for i in range(8):  # alternate kinds where possible: low thump / bright snap
        pool = (blinks, jaws)[i % 2] or blinks or jaws
        if pool:
            order.append(pool[(i // 2) % len(pool)])
    if not order:  # no events: evenly spaced grains
        order = [(eeg.shape[1] / rate * (i + 0.5) / 8, "grain") for i in range(8)]
    beat = int(round(60.0 / bpm * OUT_RATE))
    hits = np.zeros((beat * 8, 2))
    hit_len = min(beat, int(0.35 * OUT_RATE))
    for i, (et, kind) in enumerate(order):
        c = int(et * OUT_RATE / speed_h)
        start = max(0, c - int(0.02 * OUT_RATE))
        hits[i * beat: i * beat + hit_len] = _hit(aud[start:start + hit_len], hit_len)
    p = out / f"muse_hits_{bpm:g}bpm{tag}.wav"
    write_wav16(p, hits)
    made.append(p)

    (out / "README.txt").write_text(
        ("SOURCE: SIMULATED EEG (synthetic model, not a brain recording)\n" if sim else
         f"SOURCE: Muse 2 recording {folder.name}\n")
        + f"Project: key {key}, {bpm:g} BPM, {bars} bars = {seconds:.2f} s. 16-bit / 44.1 kHz.\n\n"
        + f"muse_drone_{key}: EEG wavetable ({CHANNELS[best]}) as {key}2+{key}3+fifth, scanning the recording.\n"
        + f"muse_texture: whole recording audified x{speed:.0f} (L = TP9+AF7, R = TP10+AF8), {bars} bars long.\n"
        + "muse_hits: 8 one-shots on the beat, cut from x40 audification at "
        + ", ".join(f"{k}@{t:.1f}s" for t, k in order) + ".\n"
        + "Drop all three at bar 1. Hits: Quick Sampler > Slice mode (transient) to play them from keys.\n"
    )
    for p in made:
        print(f"[stems] wrote {p.name}")
    print(f"[stems] {bars} bars @ {bpm:g} BPM = {seconds:.2f} s, key {key}; folder: {out}")
    return made
