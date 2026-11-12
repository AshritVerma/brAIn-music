"""`muse2-music` CLI.

Subcommands:
  run        Live EEG -> TUI diagnostics (legacy, no music output).
  perform    NEW Muse -> Lyria -> browser visualizer pipeline (Phase 3+).
  simulate   Synthetic brain -> TUI (headset-free dev).
  battery    Quick BLE telemetry read of the Muse 2 battery.
  bt-reset   Cycle macOS Bluetooth to recover from stuck BLE state.
  record     Start/Stop page that records EEG + Logic audio + MIDI on one clock,
             and streams EEG -> MIDI CC to Logic over the IAC bus.
  sonify     Turn a recording's EEG into WAVs (audification + wavetable).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from muse2_music_lab import config


def _default_http_port() -> int:
    """Honor PaaS-style $PORT env vars (Railway/Fly/Heroku/Render).

    Railway in particular sets $PORT to whatever the platform decided to
    route external traffic to, and refuses to route to any other port.
    Falling back to 8000 for local dev keeps the existing UX intact.
    """
    raw = os.environ.get("PORT", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            print(f"[cli] WARN: ignoring unparseable $PORT={raw!r}; using 8000")
    return 8000


# ---------------------------------------------------------------------------
# Subcommand dispatchers
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from muse2_music_lab.tui import RunOptions, run_with_tui

    opts = RunOptions(
        calibrate_seconds=args.calibrate_seconds,
        send_rate_hz=args.rate,
        window_size=args.window,
        smoothing_alpha=args.smoothing,
    )
    return run_with_tui(opts)


def _cmd_perform(args: argparse.Namespace) -> int:
    from muse2_music_lab.main import PerformOptions, run as perform_run

    opts = PerformOptions(
        prompt=args.prompt,
        http_port=args.http_port,
        no_browser=args.no_browser,
        simulate_eeg=args.simulate_eeg,
        no_lyria=args.no_lyria,
        no_server=args.no_server,
        no_tui=args.no_tui,
        skip_seed=args.skip_seed,
        no_seed_cache=args.no_seed_cache,
        evolve_chunks=args.evolve_chunks,
        cloud=args.cloud,
    )
    return perform_run(opts)


def _cmd_simulate(args: argparse.Namespace) -> int:
    from muse2_music_lab.simulate import SimulateOptions, run as sim_run

    opts = SimulateOptions(
        send_rate_hz=args.rate,
        duration_s=args.duration,
    )
    return sim_run(opts)


def _cmd_battery(_: argparse.Namespace) -> int:
    from muse2_music_lab.battery import run as battery_run

    return battery_run()


def _cmd_record(args: argparse.Namespace) -> int:
    from pathlib import Path

    from muse2_music_lab.recorder import DEFAULT_OUT_DIR, RecordOptions, run as record_run

    opts = RecordOptions(
        http_port=args.http_port,
        out_dir=Path(args.out).expanduser() if args.out else DEFAULT_OUT_DIR,
        simulate=args.simulate,
        no_browser=args.no_browser,
        audio_device=args.audio_device,
        no_audio=args.no_audio,
        no_midi=args.no_midi,
        cc_port=args.cc_port,
        cc_channel=args.cc_channel,
        no_cc=args.no_cc,
        replay=_resolve_recording(args.replay) if args.replay else None,
    )
    return record_run(opts)


def _cmd_sonify(args: argparse.Namespace) -> int:
    from pathlib import Path

    from muse2_music_lab.recorder import DEFAULT_OUT_DIR
    from muse2_music_lab.sonify import run as sonify_run

    target = args.recording
    if target in (None, "latest"):
        # newest recording that actually has samples (skip accidental Start/Stop)
        runs = sorted(p for p in DEFAULT_OUT_DIR.glob("*/eeg.csv") if p.stat().st_size > 64)
        if not runs:
            print("[sonify] no recordings yet. Record one with `muse2 record`.")
            return 1
        folder = runs[-1].parent
    else:
        folder = Path(target).expanduser()
        if not (folder / "eeg.csv").exists() and (DEFAULT_OUT_DIR / target / "eeg.csv").exists():
            folder = DEFAULT_OUT_DIR / target
    speeds = tuple(float(x) for x in args.speeds.split(",") if x.strip())
    return sonify_run(folder, speeds=speeds, frames=args.frames)


def _resolve_recording(target):
    from pathlib import Path

    from muse2_music_lab.recorder import DEFAULT_OUT_DIR

    if target in (None, "latest"):
        runs = sorted(p for p in DEFAULT_OUT_DIR.glob("*/eeg.csv") if p.stat().st_size > 64)
        return runs[-1].parent if runs else None
    folder = Path(target).expanduser()
    if not (folder / "eeg.csv").exists() and (DEFAULT_OUT_DIR / target / "eeg.csv").exists():
        folder = DEFAULT_OUT_DIR / target
    return folder


def _cmd_simulate_eeg(args: argparse.Namespace) -> int:
    from muse2_music_lab.recorder import DEFAULT_OUT_DIR
    from muse2_music_lab.stems import write_simulated_recording

    write_simulated_recording(DEFAULT_OUT_DIR, args.seconds, seed=args.seed)
    return 0


def _cmd_stems(args: argparse.Namespace) -> int:
    from pathlib import Path

    from muse2_music_lab.stems import render_stems

    folder = _resolve_recording(args.recording)
    if folder is None:
        print("[stems] no recordings yet. `muse2 record` or `muse2 simulate-eeg` first.")
        return 1
    render_stems(folder, key=args.key, bpm=args.bpm, bars=args.bars,
                 out_dir=Path(args.out).expanduser() if args.out else None)
    return 0


def _cmd_bt_reset(_: argparse.Namespace) -> int:
    from muse2_music_lab.bt_reset import run as bt_reset_run

    return bt_reset_run()


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="muse2",
        description=(
            "Stream EEG from a Muse 2 via BrainFlow. The 'perform' subcommand "
            "drives the Lyria + browser-visualizer demo; 'run' shows a TUI of "
            "the live signals; 'simulate' does the same with a synthetic brain."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ------- run (legacy EEG TUI diagnostics) -------
    pr = sub.add_parser(
        "run",
        help="Live EEG -> TUI diagnostics (no music output).",
    )
    pr.add_argument(
        "--calibrate-seconds",
        type=float,
        default=config.CALIBRATION_DURATION,
        dest="calibrate_seconds",
        help="Baseline calibration duration in seconds",
    )
    pr.add_argument(
        "--rate",
        type=float,
        default=config.SEND_RATE_HZ,
        help="TUI refresh rate in Hz",
    )
    pr.add_argument(
        "--window",
        type=int,
        default=config.WINDOW_SIZE,
        help="Samples per feature window",
    )
    pr.add_argument(
        "--smoothing",
        type=float,
        default=config.SMOOTHING_ALPHA,
        help="EMA alpha (lower = smoother, more lag)",
    )
    pr.set_defaults(func=_cmd_run)

    # ------- perform (new Lyria + visualizer pipeline) -------
    pp = sub.add_parser(
        "perform",
        help="NEW: Muse -> Lyria RealTime music + Three.js shader visualizer.",
    )
    pp.add_argument(
        "--prompt",
        default="",
        help=(
            "Session prompt. Drives both Lyria's stylistic basis and the seed "
            "image used by the visualizer (e.g. 'downtempo electronic with "
            "warm analog synth pads'). OPTIONAL since Phase 10: omit to type "
            "the prompt in the browser and click Start manually."
        ),
    )
    pp.add_argument(
        "--http-port",
        type=int,
        default=_default_http_port(),
        dest="http_port",
        help=(
            "aiohttp server port (default: 8000, or $PORT if set -- "
            "Railway and other PaaS providers pin the route to their "
            "chosen $PORT)."
        ),
    )
    pp.add_argument(
        "--no-browser",
        action="store_true",
        dest="no_browser",
        help="Skip auto-launching Chrome to the visualizer URL.",
    )
    pp.add_argument(
        "--simulate-eeg",
        action="store_true",
        dest="simulate_eeg",
        help="Use synthetic EEG instead of opening the Muse 2 (headset-free dev).",
    )
    pp.add_argument(
        "--no-lyria",
        action="store_true",
        dest="no_lyria",
        help="Skip Lyria music generation (visual + EEG only; for visualizer dev).",
    )
    pp.add_argument(
        "--no-server",
        action="store_true",
        dest="no_server",
        help="Skip the aiohttp + WebSocket server (for headless EEG/Lyria dev).",
    )
    pp.add_argument(
        "--no-tui",
        action="store_true",
        dest="no_tui",
        help=(
            "Disable the rich.Live status panel and fall back to a plain "
            "'[state] alpha=...' log line every couple of seconds. Use when "
            "piping output to a file or in a terminal that doesn't render "
            "the panel cleanly."
        ),
    )
    pp.add_argument(
        "--skip-seed",
        action="store_true",
        dest="skip_seed",
        help=(
            "Skip Phase 8's Imagen seed image call. Saves ~3-5s of startup "
            "wall time and avoids API quota usage; the visualizer will "
            "fall back to whatever static/seed.png already exists from a "
            "prior session, or render blank if there's none."
        ),
    )
    pp.add_argument(
        "--no-seed-cache",
        action="store_true",
        dest="no_seed_cache",
        help=(
            "Force re-generation of the seed image even if a cached PNG "
            "exists for this prompt. Useful when iterating on the Imagen "
            "model id or aspect ratio."
        ),
    )
    pp.add_argument(
        "--evolve-chunks",
        type=int,
        default=config.EVOLVE_INTERVAL_CHUNKS,
        dest="evolve_chunks",
        help=(
            "Phase 10: regenerate the seed image every N Lyria audio "
            "chunks (~2s of music each, so 12 chunks ≈ 24s) based on "
            "how the EEG/audio features have moved. Pass 0 to disable. "
            "Cost scales inversely with N: ~$3/hr at 12, ~$1.5/hr at 24 "
            "(default: %(default)s)."
        ),
    )
    pp.add_argument(
        "--cloud",
        action="store_true",
        dest="cloud",
        help=(
            "Public-deploy / Railway mode. Implies --simulate-eeg, "
            "--no-browser, --no-tui; binds the server to 0.0.0.0; "
            "fans Lyria audio out to every connected browser as binary "
            "WS frames instead of writing to a local sounddevice; locks "
            "per-visitor controls (Quit, EEG-mode toggle) so a single "
            "visitor can't break the experience for everyone else."
        ),
    )
    pp.set_defaults(func=_cmd_perform)

    # ------- simulate (synthetic EEG TUI) -------
    ps = sub.add_parser(
        "simulate",
        help="Synthetic brain -> TUI (headset-free smoke test of the diagnostic UI).",
    )
    ps.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="TUI refresh rate in Hz",
    )
    ps.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = run until Ctrl-C)",
    )
    ps.set_defaults(func=_cmd_simulate)

    # ------- simulate-eeg (synthetic recording) -------
    psim = sub.add_parser("simulate-eeg", help="Write a SIMULATED EEG recording (for building before real data).")
    psim.add_argument("--seconds", type=float, default=366.0)
    psim.add_argument("--seed", type=int, default=None)
    psim.set_defaults(func=_cmd_simulate_eeg)

    # ------- stems (project-ready files) -------
    pst = sub.add_parser("stems", help="Render drone / texture / hits WAVs in a project's key and tempo.")
    pst.add_argument("recording", nargs="?", default="latest")
    pst.add_argument("--key", default="C")
    pst.add_argument("--bpm", type=float, default=105.0)
    pst.add_argument("--bars", type=int, default=16)
    pst.add_argument("--out", default=None, help="Output folder (default: <recording>/stems).")
    pst.set_defaults(func=_cmd_stems)

    # ------- battery -------
    pb = sub.add_parser(
        "battery",
        help="Connect briefly to the Muse 2 and print its battery percentage.",
    )
    pb.set_defaults(func=_cmd_battery)

    # ------- record (EEG + Logic session capture) -------
    prec = sub.add_parser(
        "record",
        help=(
            "Open a Start/Stop control page and record raw EEG, features, "
            "Logic's audio (via a loopback input) and MIDI on one clock."
        ),
    )
    prec.add_argument("--http-port", type=int, default=8770, dest="http_port",
                      help="Control page port (default: %(default)s).")
    prec.add_argument("--out", default=None,
                      help="Recordings folder (default: <repo>/recordings).")
    prec.add_argument("--simulate", action="store_true",
                      help="Use BrainFlow's synthetic board instead of the Muse 2.")
    prec.add_argument("--no-browser", action="store_true", dest="no_browser",
                      help="Don't auto-open the control page.")
    prec.add_argument("--audio-device", default=None, dest="audio_device",
                      help="Audio input name (substring) to capture. Default: auto-pick BlackHole/Loopback.")
    prec.add_argument("--no-audio", action="store_true", dest="no_audio",
                      help="Don't capture audio.")
    prec.add_argument("--no-midi", action="store_true", dest="no_midi",
                      help="Don't log MIDI.")
    prec.add_argument("--cc-port", default="loopMIDI" if os.name == "nt" else "IAC Driver Bus 1", dest="cc_port",
                      help="MIDI output for EEG -> CC (default: %(default)s).")
    prec.add_argument("--cc-channel", type=int, default=1, dest="cc_channel",
                      help="MIDI channel 1-16 for CCs and trigger notes (default: %(default)s).")
    prec.add_argument("--no-cc", action="store_true", dest="no_cc",
                      help="Disable the EEG -> MIDI CC output.")
    prec.add_argument("--replay", default=None,
                      help="Stream a saved recording (folder, name, or 'latest') instead of the headset, looping.")
    prec.set_defaults(func=_cmd_record)

    # ------- sonify (EEG -> audio files) -------
    pso = sub.add_parser(
        "sonify",
        help="Turn a recording's EEG into WAVs: audification at several speeds + a wavetable.",
    )
    pso.add_argument("recording", nargs="?", default="latest",
                     help="Recording folder or its name under recordings/ (default: latest).")
    pso.add_argument("--speeds", default="10,40,172",
                     help="Comma-separated audification speed-ups (default: %(default)s).")
    pso.add_argument("--frames", type=int, default=64,
                     help="Wavetable frames (default: %(default)s).")
    pso.set_defaults(func=_cmd_sonify)

    # ------- bt-reset -------
    pbt = sub.add_parser(
        "bt-reset",
        help=(
            "Cycle macOS Bluetooth to recover from a stuck BLE state "
            "(use after a Ctrl-C-killed run that won't reconnect)."
        ),
    )
    pbt.set_defaults(func=_cmd_bt_reset)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    raise SystemExit(ns.func(ns))
