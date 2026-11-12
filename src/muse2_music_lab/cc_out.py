"""EEG features -> MIDI CC (and trigger notes) over a virtual MIDI port.

Default route is macOS's IAC Driver bus: Logic sees it as a MIDI input, and
any CC can be bound to any plugin parameter with Logic's Learn.

Design choices that make it playable rather than jittery:

* Self-calibrating range. Each feature is mapped through its own rolling
  5th..95th percentile over the last ``window_s`` seconds, so there is no
  calibration step and the full 0..127 range stays in use as your baseline
  drifts across a session.
* Smoothing + upsampling. Features arrive at 4 Hz; CCs go out at ~30 Hz
  through a one-pole smoother, so automation glides instead of stepping.
* Learn mode. While a feature is "learning", every other CC is muted and
  that one sweeps 0->127->0, so Logic's Learn binds exactly that controller.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

DEFAULT_PORT = "IAC Driver Bus 1"

# feature -> CC number. 20-31 are "undefined" in the MIDI spec, so they won't
# collide with mod wheel (1), volume (7), sustain (64) etc.
DEFAULT_CC_MAP: dict[str, int] = {
    "alpha": 20,
    "beta": 21,
    "theta": 22,
    "focus": 23,
    "calm": 24,
}
# discrete events -> note numbers (C3 / D3 in Logic's naming)
DEFAULT_NOTE_MAP: dict[str, int] = {"blink": 60, "jaw": 62}

SEND_HZ = 30.0
LEARN_SECONDS = 4.0


@dataclass
class CCBridge:
    get_features: Callable[[], dict[str, Any]]
    log: Callable[[float, str, int, int], None] = lambda *a: None
    port_name: str = DEFAULT_PORT
    channel: int = 0                     # 0-based MIDI channel (0 = ch 1)
    cc_map: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CC_MAP))
    note_map: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_NOTE_MAP))
    window_s: float = 30.0
    glide: float = 0.15                  # one-pole coefficient per 30 Hz tick
    enabled: bool = False
    notes_enabled: bool = False
    error: str = ""
    learning: Optional[str] = None
    values: dict[str, int] = field(default_factory=dict)
    _hist: dict[str, deque] = field(default_factory=dict)
    _smooth: dict[str, float] = field(default_factory=dict)
    _last_sent: dict[str, int] = field(default_factory=dict)
    _last_feat_t: float = 0.0
    _learn_until: float = 0.0
    _port: Any = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        threading.Thread(target=self._loop, name="cc-out", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._close()

    def _open(self) -> bool:
        if self._port is not None:
            return True
        try:
            import mido

            names = mido.get_output_names()
            match = next((n for n in names if self.port_name.lower() in n.lower()), None)
            if match is None:
                self.error = (
                    f"MIDI port '{self.port_name}' not found (have: {', '.join(names) or 'none'}). "
                    "Enable the IAC Driver in Audio MIDI Setup > MIDI Studio."
                )
                return False
            self._port = mido.open_output(match)
            self.port_name = match
            self.error = ""
            return True
        except Exception as e:  # pragma: no cover - device dependent
            self.error = f"could not open MIDI output: {e}"
            return False

    def _close(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None

    def set_enabled(self, on: bool) -> dict[str, Any]:
        with self._lock:
            self.enabled = bool(on)
            if self.enabled and not self._open():
                self.enabled = False
                return {"ok": False, "error": self.error}
        return {"ok": True, "enabled": self.enabled}

    def set_notes(self, on: bool) -> dict[str, Any]:
        self.notes_enabled = bool(on)
        return {"ok": True, "notes": self.notes_enabled}

    def learn(self, feature: str) -> dict[str, Any]:
        if feature not in self.cc_map:
            return {"ok": False, "error": f"unknown feature {feature!r}"}
        if not self._open():
            return {"ok": False, "error": self.error}
        self.learning = feature
        self._learn_until = time.time() + LEARN_SECONDS
        return {"ok": True, "learning": feature, "cc": self.cc_map[feature], "seconds": LEARN_SECONDS}

    # ---------------- core ----------------

    def _normalize(self, name: str, x: float) -> float:
        h = self._hist[name]
        if len(h) < 8:
            return 0.5
        lo, hi = np.percentile(np.asarray(h), [5, 95])
        if hi - lo < 1e-9:
            return 0.5
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def _send_cc(self, name: str, value: int, now: float) -> None:
        import mido

        cc = self.cc_map[name]
        self._port.send(mido.Message("control_change", channel=self.channel, control=cc, value=value))
        self._last_sent[name] = value
        self.log(now, name, cc, value)

    def _send_note(self, name: str, now: float) -> None:
        import mido

        note = self.note_map[name]
        self._port.send(mido.Message("note_on", channel=self.channel, note=note, velocity=100))
        self.log(now, f"{name}_note", note, 100)

        def _off(p=self._port, n=note):
            time.sleep(0.15)
            try:
                p.send(mido.Message("note_off", channel=self.channel, note=n, velocity=0))
            except Exception:
                pass

        threading.Thread(target=_off, daemon=True).start()

    def _loop(self) -> None:
        period = 1.0 / SEND_HZ
        maxlen = int(self.window_s * 4) + 4
        while not self._stop.is_set():
            t_next = time.time() + period
            try:
                self._tick(maxlen)
            except Exception as e:  # keep the loop alive; surface the error
                self.error = f"CC loop: {e!r}"
                self._close()
            time.sleep(max(0.0, t_next - time.time()))

    def _tick(self, maxlen: int) -> None:
        now = time.time()
        feats = self.get_features() or {}
        fresh = feats.get("t", 0.0) > self._last_feat_t
        if fresh:
            self._last_feat_t = feats["t"]
            for name in self.cc_map:
                if name in feats:
                    self._hist.setdefault(name, deque(maxlen=maxlen)).append(float(feats[name]))

        # targets in 0..1
        for name in self.cc_map:
            if name not in self._hist:
                continue
            target = self._normalize(name, self._hist[name][-1])
            prev = self._smooth.get(name, target)
            self._smooth[name] = prev + self.glide * (target - prev)
            self.values[name] = int(round(self._smooth[name] * 127))

        if self._port is None:
            return

        # learn mode: only the learning CC, as a clean triangle sweep
        if self.learning:
            remaining = self._learn_until - now
            if remaining <= 0:
                self.learning = None
            else:
                phase = 1.0 - remaining / LEARN_SECONDS        # 0..1
                tri = 1.0 - abs(2.0 * ((phase * 2.0) % 1.0) - 1.0)  # two up/down sweeps
                v = int(round(tri * 127))
                if self._last_sent.get(self.learning) != v:
                    self._send_cc(self.learning, v, now)
                return

        if not self.enabled:
            return

        for name, v in self.values.items():
            if self._last_sent.get(name) != v:
                self._send_cc(name, v, now)

        if self.notes_enabled and fresh:
            for name in self.note_map:
                if feats.get(name):
                    self._send_note(name, now)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "notes": self.notes_enabled,
            "port": self.port_name,
            "channel": self.channel + 1,
            "error": self.error,
            "learning": self.learning,
            "map": [
                {"feature": k, "cc": v, "value": self.values.get(k)} for k, v in self.cc_map.items()
            ],
            "note_map": self.note_map,
        }
