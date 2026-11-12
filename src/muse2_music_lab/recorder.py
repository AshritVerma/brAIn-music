"""`muse2 record` — capture EEG alongside a Logic Pro session.

One process holds the Muse 2 BLE link open the whole time (so Start/Stop
never has to fight CoreBluetooth), serves a small control page, and on
Start writes everything below into ``recordings/<timestamp>[_label]/``
on a single clock (host Unix time, the same clock BrainFlow stamps
Muse samples with):

    eeg.csv        raw EEG, 256 Hz: unix_time, package_num, TP9, AF7, AF8, TP10
    motion.csv     accelerometer + gyro (if the headset streams them)
    features.csv   4 Hz alpha/beta/theta/focus/calm/blink/jaw (same math
                   as `muse2 perform`, so later real-time modulation can
                   be trained/tuned on exactly these numbers)
    audio.wav      what Logic plays, via a loopback input (BlackHole etc.)
    audio_sync.csv frame index <-> unix time, ~2x/second (drift-proof alignment)
    midi.csv       every MIDI message on the selected inputs (your keyboard,
                   Logic's MIDI clock / MMC on the IAC bus, ...)
    markers.csv    manual markers from the page (M key / Mark button)
    events.csv     connect / disconnect / start / stop log
    session.json   summary + device info, written on Stop

Nothing is written while not recording; the live view keeps running so
you can check contact quality before you hit Start.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import queue
import re
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from muse2_music_lab import config

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "recordings"

POLL_S = 0.1                 # BrainFlow drain cadence
FEATURE_PERIOD_S = 0.25      # 4 Hz, matches perform pipeline
LIVE_SECONDS = 4.0           # rolling buffer kept for the live view
AUDIO_SYNC_EVERY_S = 0.5


@dataclass
class RecordOptions:
    http_port: int = 8770
    out_dir: Path = DEFAULT_OUT_DIR
    simulate: bool = False
    no_browser: bool = False
    audio_device: Optional[str] = None   # name substring; None = auto (BlackHole/Loopback)
    no_audio: bool = False
    no_midi: bool = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-")
    return s[:60]


class _CsvSink:
    """Append-only CSV writer that flushes on every write batch."""

    def __init__(self, path: Path, header: list[str]) -> None:
        self.path = path
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(header)
        self.rows = 0

    def write_rows(self, rows) -> None:
        n = 0
        for r in rows:
            self._w.writerow(r)
            n += 1
        if n:
            self.rows += n
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class _Wav24Writer:
    """Minimal 24-bit PCM WAV writer (stdlib `wave`, no extra deps)."""

    def __init__(self, path: Path, sample_rate: int, channels: int) -> None:
        import wave

        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(channels)
        self._w.setsampwidth(3)
        self._w.setframerate(sample_rate)
        self.frames = 0

    def write(self, block: np.ndarray) -> None:
        x = np.clip(block, -1.0, 1.0)
        i32 = (x * 8388607.0).astype("<i4")
        b = i32.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        self._w.writeframes(b)
        self.frames += block.shape[0]

    def close(self) -> None:
        try:
            self._w.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Audio (loopback input from Logic)
# ---------------------------------------------------------------------------


def list_audio_inputs() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": d["name"],
                    "channels": int(d["max_input_channels"]),
                    "rate": int(d.get("default_samplerate") or 48000),
                }
            )
    return out


def pick_default_audio(inputs: list[dict[str, Any]], wanted: Optional[str]) -> Optional[str]:
    names = [d["name"] for d in inputs]
    if wanted:
        for n in names:
            if wanted.lower() in n.lower():
                return n
        return None
    for key in ("blackhole", "loopback", "soundflower", "aggregate"):
        for n in names:
            if key in n.lower():
                return n
    return None


class AudioCapture:
    """Records a sounddevice input to a WAV + a sync table, off the RT thread."""

    def __init__(self, device_name: str, folder: Path, log_event) -> None:
        import sounddevice as sd

        self._sd = sd
        info = sd.query_devices(device_name, "input")
        self.device_name = info["name"]
        self.rate = int(info.get("default_samplerate") or 48000)
        self.channels = min(2, int(info["max_input_channels"])) or 1
        self._q: "queue.Queue[tuple[np.ndarray, float, float]]" = queue.Queue()
        self._wav = _Wav24Writer(folder / "audio.wav", self.rate, self.channels)
        self._sync = _CsvSink(folder / "audio_sync.csv", ["frame_index", "unix_time", "adc_time"])
        self._log_event = log_event
        self._stop = threading.Event()
        self._last_sync = 0.0
        self.peak = 0.0
        self.overflows = 0
        self.start_unix: Optional[float] = None
        self._stream = sd.InputStream(
            device=device_name,
            channels=self.channels,
            samplerate=self.rate,
            dtype="float32",
            callback=self._callback,
        )
        self._writer = threading.Thread(target=self._drain, name="audio-writer", daemon=True)

    def _callback(self, indata, frames, time_info, status) -> None:  # RT thread
        if status and status.input_overflow:
            self.overflows += 1
        now = time.time()
        # Unix time of the FIRST sample in this block ≈ now - (current - adc).
        try:
            latency = float(time_info.currentTime - time_info.inputBufferAdcTime)
            adc = float(time_info.inputBufferAdcTime)
        except Exception:
            latency, adc = 0.0, 0.0
        self._q.put((indata.copy(), now - max(latency, 0.0), adc))

    def _drain(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                block, t0, adc = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.start_unix is None:
                self.start_unix = t0
            if t0 - self._last_sync >= AUDIO_SYNC_EVERY_S:
                self._sync.write_rows([[self._wav.frames, f"{t0:.6f}", f"{adc:.6f}"]])
                self._last_sync = t0
            self._wav.write(block)
            self.peak = max(self.peak * 0.85, float(np.max(np.abs(block))) if block.size else 0.0)

    def start(self) -> None:
        self._writer.start()
        self._stream.start()

    def stop(self) -> dict[str, Any]:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stop.set()
        self._writer.join(timeout=5)
        self._wav.close()
        self._sync.close()
        return {
            "device": self.device_name,
            "sample_rate": self.rate,
            "channels": self.channels,
            "frames": self._wav.frames,
            "seconds": round(self._wav.frames / float(self.rate), 3),
            "first_sample_unix": self.start_unix,
            "overflows": self.overflows,
        }


# ---------------------------------------------------------------------------
# MIDI (keyboard + Logic clock/MMC over IAC)
# ---------------------------------------------------------------------------


def list_midi_inputs() -> list[str]:
    try:
        import mido

        return list(mido.get_input_names())
    except Exception:
        return []


class MidiCapture:
    def __init__(self, port_names: list[str], folder: Path) -> None:
        import mido

        self._sink = _CsvSink(folder / "midi.csv", ["unix_time", "port", "type", "channel", "data1", "data2", "raw_hex"])
        self._lock = threading.Lock()
        self._pending: list[list[Any]] = []
        self._ports = []
        self.count = 0
        for name in port_names:
            try:
                self._ports.append(mido.open_input(name, callback=self._make_cb(name)))
            except Exception as e:  # pragma: no cover - device dependent
                print(f"[record] MIDI: could not open {name!r}: {e}")

    def _make_cb(self, name: str):
        def _cb(msg) -> None:
            t = time.time()
            ch = getattr(msg, "channel", "")
            d1 = getattr(msg, "note", getattr(msg, "control", getattr(msg, "pos", getattr(msg, "program", ""))))
            d2 = getattr(msg, "velocity", getattr(msg, "value", getattr(msg, "pitch", "")))
            if msg.type == "quarter_frame":
                d1, d2 = msg.frame_type, msg.frame_value
            row = [f"{t:.6f}", name, msg.type, ch, d1, d2, msg.hex()]
            with self._lock:
                self._pending.append(row)

        return _cb

    def flush(self) -> None:
        with self._lock:
            rows, self._pending = self._pending, []
        self._sink.write_rows(rows)
        self.count += len(rows)

    def stop(self) -> dict[str, Any]:
        for p in self._ports:
            try:
                p.close()
            except Exception:
                pass
        self.flush()
        self._sink.close()
        return {"ports": [p.name for p in self._ports], "messages": self.count}


# ---------------------------------------------------------------------------
# Recorder core
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    folder: Path
    label: str
    started_unix: float
    eeg: _CsvSink
    motion: Optional[_CsvSink]
    features: _CsvSink
    markers: _CsvSink
    audio: Optional[AudioCapture] = None
    midi: Optional[MidiCapture] = None
    eeg_gap_s: float = 0.0
    last_eeg_t: Optional[float] = None


@dataclass
class Recorder:
    opts: RecordOptions
    status: str = "starting"
    status_detail: str = ""
    sampling_rate: int = 256
    session: Optional[_Session] = None
    last_summary: Optional[dict[str, Any]] = None
    audio_device: Optional[str] = None
    midi_ports: list[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _stop: threading.Event = field(default_factory=threading.Event)
    _live: Optional[np.ndarray] = None       # (4, n) rolling EEG
    _live_t: Optional[np.ndarray] = None
    _latest_features: dict[str, Any] = field(default_factory=dict)
    _events: list[list[Any]] = field(default_factory=list)
    _motion_ok: Optional[bool] = None

    # ---------------- lifecycle ----------------

    def start_background(self) -> None:
        self.opts.out_dir.mkdir(parents=True, exist_ok=True)
        inputs = list_audio_inputs()
        if not self.opts.no_audio:
            self.audio_device = pick_default_audio(inputs, self.opts.audio_device)
        if not self.opts.no_midi:
            self.midi_ports = list_midi_inputs()
        threading.Thread(target=self._acquire_guarded, name="eeg-acquire", daemon=True).start()

    def shutdown(self) -> None:
        if self.session is not None:
            self.stop_recording()
        self._stop.set()

    def _event(self, kind: str, detail: str = "") -> None:
        row = [f"{time.time():.6f}", kind, detail]
        print(f"[record] {kind} {detail}".rstrip(), flush=True)
        with self._lock:
            self._events.append(row)

    # ---------------- EEG acquisition thread ----------------

    def _make_board(self):
        from muse2_music_lab.eeg.board import Board

        if self.opts.simulate:
            from brainflow.board_shim import BoardIds

            return Board(board_id=BoardIds.SYNTHETIC_BOARD.value)
        return Board()

    def _acquire_guarded(self) -> None:
        try:
            self._acquire_loop()
        except Exception as e:  # never die silently: surface it on the page
            import traceback

            traceback.print_exc()
            self.status = "error"
            self.status_detail = f"EEG thread crashed: {e!r}. Restart `muse2 record`."
            self._event("crash", repr(e))

    def _acquire_loop(self) -> None:
        from brainflow.board_shim import BoardShim, BrainFlowPresets
        from brainflow.exit_codes import BrainFlowError

        from muse2_music_lab.eeg.features import BlinkDetector, JawClenchDetector, compute_frame

        backoff = config.EEG_RECONNECT_BACKOFF_S
        while not self._stop.is_set():
            board = self._make_board()
            self.status = "connecting"
            self.status_detail = "simulated board" if self.opts.simulate else "looking for Muse 2 over Bluetooth..."
            try:
                board.start()
            except Exception as e:
                self.status = "disconnected"
                self.status_detail = f"connect failed: {e}. Retrying in {backoff:.0f}s (is the headset on and free?)"
                self._event("connect_failed", str(e))
                self._stop.wait(backoff)
                continue

            bid = board.board_id
            shim = board._shim
            self.sampling_rate = BoardShim.get_sampling_rate(bid)
            eeg_rows = BoardShim.get_eeg_channels(bid)[:4]
            ts_row = BoardShim.get_timestamp_channel(bid)
            try:
                pkg_row = BoardShim.get_package_num_channel(bid)
            except Exception:
                pkg_row = None
            try:
                aux = BrainFlowPresets.AUXILIARY_PRESET
                acc_rows = BoardShim.get_accel_channels(bid, aux)
                gyro_rows = BoardShim.get_gyro_channels(bid, aux)
                aux_ts = BoardShim.get_timestamp_channel(bid, aux)
                motion_supported = True
            except Exception:
                motion_supported = False
            if self.opts.simulate:
                motion_supported = False  # synthetic board has no aux preset

            self.status = "connected"
            self.status_detail = "simulated EEG" if self.opts.simulate else "Muse 2 streaming"
            self._event("connected", self.status_detail)
            blink = BlinkDetector()
            jaw = JawClenchDetector(self.sampling_rate)
            n_live = int(LIVE_SECONDS * self.sampling_rate)
            self._live = np.zeros((4, 0))
            self._live_t = np.zeros(0)
            next_feat = time.time()
            last_data = time.time()

            try:
                while not self._stop.is_set():
                    time.sleep(POLL_S)
                    data = shim.get_board_data()
                    now = time.time()
                    if data.shape[1]:
                        last_data = now
                        eeg = np.asarray(data[eeg_rows, :], dtype=np.float64)
                        ts = np.asarray(data[ts_row, :], dtype=np.float64)
                        pk = data[pkg_row, :] if pkg_row is not None else np.zeros(data.shape[1])
                        with self._lock:
                            self._live = np.concatenate([self._live, eeg], axis=1)[:, -n_live:]
                            self._live_t = np.concatenate([self._live_t, ts])[-n_live:]
                            s = self.session
                            if s is not None:
                                if s.last_eeg_t is not None and ts[0] - s.last_eeg_t > 0.1:
                                    s.eeg_gap_s += float(ts[0] - s.last_eeg_t)
                                s.last_eeg_t = float(ts[-1])
                                s.eeg.write_rows(
                                    [f"{ts[i]:.6f}", int(pk[i])] + [f"{v:.3f}" for v in eeg[:, i]]
                                    for i in range(eeg.shape[1])
                                )
                    elif now - last_data > 5.0:
                        raise ConnectionError("no EEG samples for 5 s (headset off / out of range?)")

                    if motion_supported:
                        try:
                            m = shim.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)
                            if m.shape[1]:
                                self._motion_ok = True
                                with self._lock:
                                    s = self.session
                                    if s is not None and s.motion is not None:
                                        rows = list(acc_rows) + list(gyro_rows)
                                        s.motion.write_rows(
                                            [f"{m[aux_ts, i]:.6f}"] + [f"{m[r, i]:.5f}" for r in rows]
                                            for i in range(m.shape[1])
                                        )
                        except BrainFlowError:
                            motion_supported = False
                            self._motion_ok = False

                    if now >= next_feat:
                        next_feat = now + FEATURE_PERIOD_S
                        with self._lock:
                            win = self._live[:, -config.WINDOW_SIZE:]
                        if win.shape[1] >= config.WINDOW_SIZE:  # full window: welch needs a power-of-two length
                            fr = compute_frame(win, self.sampling_rate, blink, jaw)
                            feats = fr.as_dict()
                            feats["t"] = now
                            self._latest_features = feats
                            with self._lock:
                                s = self.session
                                if s is not None:
                                    s.features.write_rows(
                                        [[
                                            f"{now:.6f}",
                                            *(f"{feats[k]:.5f}" for k in ("alpha", "beta", "theta", "focus", "calm")),
                                            int(feats["blink"]),
                                            int(feats["jaw"]),
                                        ]]
                                    )
                    with self._lock:
                        s = self.session
                        if s is not None and s.midi is not None:
                            s.midi.flush()
            except Exception as e:
                self._event("disconnected", str(e))
                self.status = "reconnecting"
                self.status_detail = f"lost the headset ({e}). Reconnecting... recording keeps going."
            finally:
                try:
                    board.stop()
                except Exception:
                    pass
            if not self._stop.is_set():
                self._stop.wait(backoff)

    # ---------------- recording control ----------------

    def start_recording(self, label: str = "", audio_device: Optional[str] = None, midi_ports: Optional[list[str]] = None) -> dict[str, Any]:
        with self._lock:
            if self.session is not None:
                return {"ok": False, "error": "already recording"}
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = stamp + (f"_{_slug(label)}" if label.strip() else "")
            folder = self.opts.out_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            names = list(config.EEG_CHANNEL_NAMES)
            sess = _Session(
                folder=folder,
                label=label.strip(),
                started_unix=time.time(),
                eeg=_CsvSink(folder / "eeg.csv", ["unix_time", "package_num", *names]),
                motion=None
                if self.opts.simulate
                else _CsvSink(folder / "motion.csv", ["unix_time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]),
                features=_CsvSink(folder / "features.csv", ["unix_time", "alpha", "beta", "theta", "focus", "calm", "blink", "jaw"]),
                markers=_CsvSink(folder / "markers.csv", ["unix_time", "elapsed_s", "text"]),
            )
            self._events = []
            self.session = sess

        warnings = []
        dev = audio_device if audio_device is not None else self.audio_device
        if dev and not self.opts.no_audio:
            try:
                sess.audio = AudioCapture(dev, folder, self._event)
                sess.audio.start()
                self.audio_device = dev
            except Exception as e:
                warnings.append(f"audio not recording: {e}")
                sess.audio = None
        ports = midi_ports if midi_ports is not None else self.midi_ports
        if ports and not self.opts.no_midi:
            try:
                sess.midi = MidiCapture(ports, folder)
            except Exception as e:
                warnings.append(f"MIDI not recording: {e}")
        self._event("start", str(folder))
        for w in warnings:
            self._event("warning", w)
        return {"ok": True, "folder": str(folder), "warnings": warnings}

    def add_marker(self, text: str) -> dict[str, Any]:
        with self._lock:
            s = self.session
            if s is None:
                return {"ok": False, "error": "not recording"}
            t = time.time()
            s.markers.write_rows([[f"{t:.6f}", f"{t - s.started_unix:.3f}", text.strip()]])
            return {"ok": True, "count": s.markers.rows}

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            s = self.session
            if s is None:
                return {"ok": False, "error": "not recording"}
            self.session = None
        stopped = time.time()
        audio = s.audio.stop() if s.audio else None
        midi = s.midi.stop() if s.midi else None
        for sink in (s.eeg, s.motion, s.features, s.markers):
            if sink is not None:
                sink.close()
        if s.motion is not None and s.motion.rows == 0:
            try:
                s.motion.path.unlink()
            except Exception:
                pass
        self._event("stop", str(s.folder))
        with self._lock:
            events = list(self._events)
        ev = _CsvSink(s.folder / "events.csv", ["unix_time", "event", "detail"])
        ev.write_rows(events)
        ev.close()
        dur = stopped - s.started_unix
        summary = {
            "folder": str(s.folder),
            "label": s.label,
            "started_unix": s.started_unix,
            "stopped_unix": stopped,
            "started_local": datetime.fromtimestamp(s.started_unix).isoformat(timespec="seconds"),
            "duration_s": round(dur, 3),
            "clock": "All *_time / unix_time columns are host Unix seconds (time.time()); BrainFlow stamps Muse samples on the same clock.",
            "eeg": {
                "source": "synthetic" if self.opts.simulate else "Muse 2 (BrainFlow MUSE_2_BOARD)",
                "sampling_rate": self.sampling_rate,
                "channels": list(config.EEG_CHANNEL_NAMES),
                "units": "microvolts",
                "samples": s.eeg.rows,
                "effective_rate": round(s.eeg.rows / dur, 2) if dur > 0 else None,
                "gap_seconds": round(s.eeg_gap_s, 3),
            },
            "motion_samples": s.motion.rows if s.motion else 0,
            "feature_frames": s.features.rows,
            "markers": s.markers.rows,
            "audio": audio,
            "midi": midi,
        }
        (s.folder / "session.json").write_text(json.dumps(summary, indent=2))
        self.last_summary = summary
        return {"ok": True, "summary": summary}

    # ---------------- live view ----------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            live = None if self._live is None else self._live.copy()
            s = self.session
        quality = []
        trace = []
        if live is not None and live.shape[1] > 0:
            one_s = live[:, -self.sampling_rate:]
            for ch in range(live.shape[0]):
                x = one_s[ch] - np.mean(one_s[ch])
                sd = float(np.std(x))
                if sd < 0.5:
                    q = "flat"
                elif sd > 150:
                    q = "noisy"
                elif sd > 60:
                    q = "fair"
                else:
                    q = "good"
                quality.append({"name": config.EEG_CHANNEL_NAMES[ch], "std_uv": round(sd, 1), "q": q})
            two = live[:, -2 * self.sampling_rate:]
            step = max(1, two.shape[1] // 200)
            dec = two[:, ::step]
            dec = dec - np.mean(dec, axis=1, keepdims=True)
            trace = np.round(dec, 1).tolist()
        rec = None
        if s is not None:
            rec = {
                "folder": s.folder.name,
                "elapsed_s": round(time.time() - s.started_unix, 1),
                "eeg_samples": s.eeg.rows,
                "markers": s.markers.rows,
                "audio": bool(s.audio),
                "audio_peak": round(s.audio.peak, 3) if s.audio else 0.0,
                "midi_messages": (s.midi.count if s.midi else 0),
            }
        feats = {k: v for k, v in self._latest_features.items() if k != "t"}
        return {
            "status": self.status,
            "detail": self.status_detail,
            "simulate": self.opts.simulate,
            "quality": quality,
            "trace": trace,
            "features": feats,
            "recording": rec,
            "last_summary": self.last_summary,
            "audio_device": self.audio_device,
            "midi_ports": self.midi_ports,
            "out_dir": str(self.opts.out_dir),
        }


# ---------------------------------------------------------------------------
# HTTP control page
# ---------------------------------------------------------------------------


def _build_app(rec: Recorder):
    from aiohttp import web

    async def index(_req):
        return web.FileResponse(STATIC_DIR / "record.html")

    async def state(_req):
        return web.json_response(rec.snapshot())

    async def devices(_req):
        return web.json_response(
            {
                "audio_inputs": [d["name"] for d in list_audio_inputs()],
                "midi_inputs": list_midi_inputs(),
                "audio_device": rec.audio_device,
                "midi_ports": rec.midi_ports,
            }
        )

    async def _body(req) -> dict[str, Any]:
        try:
            return await req.json()
        except Exception:
            return {}

    async def start(req):
        b = await _body(req)
        res = await asyncio.to_thread(
            rec.start_recording,
            str(b.get("label", "")),
            (str(b.get("audio_device") or "") if "audio_device" in b else None),
            b.get("midi_ports", None),
        )
        return web.json_response(res)

    async def stop(_req):
        return web.json_response(await asyncio.to_thread(rec.stop_recording))

    async def mark(req):
        b = await _body(req)
        return web.json_response(rec.add_marker(str(b.get("text", ""))))

    async def reveal(_req):
        target = rec.last_summary["folder"] if rec.last_summary else str(rec.opts.out_dir)
        if os.uname().sysname == "Darwin":
            os.system(f"open {json.dumps(target)}")
        return web.json_response({"ok": True, "folder": target})

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/state", state)
    app.router.add_get("/devices", devices)
    app.router.add_post("/start", start)
    app.router.add_post("/stop", stop)
    app.router.add_post("/mark", mark)
    app.router.add_post("/reveal", reveal)
    return app


def run(opts: RecordOptions) -> int:
    from aiohttp import web

    rec = Recorder(opts)
    rec.start_background()
    url = f"http://localhost:{opts.http_port}/"
    print(f"[record] control page: {url}")
    print(f"[record] recordings go to: {opts.out_dir}")
    print(f"[record] audio input: {rec.audio_device or 'none (install BlackHole to capture Logic)'}")
    print(f"[record] MIDI inputs: {', '.join(rec.midi_ports) or 'none'}")
    print("[record] Ctrl-C to quit (an in-progress recording is saved first).")

    import signal

    def _term(*_):  # treat SIGTERM / closed-terminal like Ctrl-C so the session is saved
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _term)
        except (ValueError, AttributeError, OSError):
            pass

    async def _main() -> None:
        runner = web.AppRunner(_build_app(rec))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", opts.http_port)
        await site.start()
        if not opts.no_browser:
            webbrowser.open(url)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    finally:
        rec.shutdown()
        time.sleep(0.3)
        if rec.last_summary:
            print(f"[record] last session saved: {rec.last_summary['folder']}")
    return 0
