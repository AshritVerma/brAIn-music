# brAIn-music

Live music + visuals driven by an EEG headset.

A single Python process reads EEG from an **Interaxon Muse 2** over
Bluetooth, generates **continuous music** with **Google Lyria RealTime**,
and renders **continuous visuals** in a Chromium tab via a **Three.js
+ GLSL shader**, all modulated in real time by what your brain is
doing. One text prompt at session start drives both the music and the
visuals.

```
+----------+     +-----------+     +----------------+     +------------+
| Muse 2   | --> | EEG loop  | --> |   AppState     | --> | Lyria      |
| (BLE)    |     | (4 Hz)    |     |  (asyncio)     |     | RealTime   |
+----------+     +-----------+     +-------+--------+     +-----+------+
                                           |                    |
                                           v                    v
                                   +---------------+    +---------------+
                                   | HTTP/WS srv   |    |  sounddevice  |
                                   | (20 Hz snap)  |    |   playback    |
                                   +-------+-------+    +-------+-------+
                                           |                    |
                                           v                    v
                                   +-------+-------+    +-------+-------+
                                   |  Browser UI   |    |   Audio FFT   |
                                   |  + Three.js   | <--+ (RMS/cent/    |
                                   |    shader     |    |    onset)     |
                                   +---------------+    +---------------+
```

No DAW. No MIDI. No TouchDesigner. No multi-process IPC. Open the
browser tab, type a prompt, hit Start, put the headset on.

## What you need

- **Hardware**
  - Interaxon **Muse 2** EEG headset, charged and powered on
  - macOS with Bluetooth (Apple Silicon recommended; tested on M4 Max)
  - Speakers or headphones (Lyria streams 48 kHz stereo)
- **Accounts / API keys**
  - **Gemini API key** with access to `lyria-realtime-exp` and
    `imagen-4.0-fast-generate-001` (Google AI Studio)
  - *(Optional but recommended)* **Anthropic API key** for the
    Claude-Opus prompt-guard and seed-evolver polishing
- **Software**
  - **Python 3.11** (the `[tool.uv]` venv pins this)
  - A **Chromium-family browser** (Chrome / Edge / Brave) for the visualizer

## Setup

```bash
git clone https://github.com/AshritVerma/brAIn-music.git
cd brAIn-music

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .

cp .env.example .env
# edit .env and fill in:
#   GEMINI_API_KEY=...
#   ANTHROPIC_API_KEY=...   (optional)
```

First-run note on macOS Bluetooth: the OS will prompt your terminal
(or your editor's integrated terminal) for Bluetooth access the first
time BrainFlow tries to scan. Approve it.

## Usage

### The full pipeline

```bash
muse2 perform
```

What happens:

1. The HTTP/WS server starts on `http://localhost:8765/`.
2. The browser tab opens automatically.
3. The EEG supervisor connects to the Muse 2 (or starts the simulator
   if `--simulate-eeg` was passed).
4. You type a text prompt in the centered Start panel and hit **Start ↵**.
5. Imagen generates a seed image from the prompt; Lyria opens a streaming
   session and begins emitting 48 kHz stereo PCM; the browser cross-fades
   from the placeholder to the real seed.
6. The visualizer modulates the seed in real time using your brain
   features (alpha/beta/theta/asymmetry) and the music's audio features
   (RMS / spectral centroid / onset). Every ~24 s of music the seed
   evolves into a new image based on where your brain has been drifting.

Browser controls (always available):

- **Start** — begins the Lyria session and seed image generation
- **Click the prompt at the top** — change the prompt mid-session
  without restarting. Type a new prompt + Enter; Lyria crossfades
  to it over ~16 s by ramping a `set_weighted_prompts` blend from
  `(old=1.0, new=0.0)` to `(old=0.0, new=1.0)`. A pulsing badge
  next to the prompt shows live `→ <new prompt> (NN%)` progress.
  Type a second prompt mid-crossfade to redirect (latest-wins).
  Escape (or click away) cancels back to display mode.
- **Recalibrate** — rebaselines the EEG normalizer in place. While
  the 8 s baseline window is open a floating amber banner shows
  *"calibrating EEG — sit still, eyes open"* with a live countdown
  and progress bar so you know exactly how long until the music
  starts responding to you again. Same banner appears on initial
  connect for the very first calibration. The Recalibrate button
  is disabled until the window closes
- **Quit** — stops every task and exits the Python process cleanly
- **EEG: real / simulated** — hot-swap between the headset and the
  synthetic generator without restarting
- **Tune** — opens a right-side drawer for live threshold tuning.
  Adjust the blink trigger (µV peak-to-peak), jaw clench trigger
  (µV peak |HP|), per-band sensitivity multipliers for alpha / beta
  / theta (rescue any band that has drifted into permanent 0/1
  saturation without recalibrating), and the Lyria sensitivity gain
  on the fly. Each row has a slider + a number input + a reset
  button, plus a live peak meter so you can see the current EEG
  reading next to the threshold marker (or the post-multiplier
  normalized value, for sensitivity rows). Hidden in `--cloud` mode
  (a shared visitor must not be able to retune for everyone else)
- **Record / Save** — capture exactly what's playing (visualizer
  canvas + Lyria audio) into a single `.webm` file (VP9 video +
  Opus audio). Click Record to start (a red dot pulses + an elapsed
  timer ticks), click again to stop, then Save to download. Works
  in both local mode (sounddevice still owns the speakers; the
  browser captures the same PCM stream silently for the recording)
  and cloud mode. The file lands in your Downloads folder named
  `brAIn-music_<YYYY-MM-DD_HH-mm-ss>.webm`.

### Useful flags

```bash
# Drive the entire pipeline off the synthetic EEG (no headset needed)
muse2 perform --simulate-eeg

# Visual-only smoke test (no audio bills, no Lyria latency)
muse2 perform --simulate-eeg --no-lyria

# Reuse the previous session's seed image instead of regenerating
muse2 perform --skip-seed

# Change how often the seed image evolves (default 12 chunks ~= 24s)
muse2 perform --evolve-chunks 24

# Skip the rich TUI (useful when piping logs)
muse2 perform --no-tui

# Pre-fill the prompt at the CLI; auto-starts (no Start panel)
muse2 perform --prompt "downtempo electronic with warm analog pads"
```

### EEG-only diagnostic TUI

```bash
muse2 run                # real Muse 2
muse2 simulate           # synthetic EEG
```

Live `rich` TUI showing band powers, asymmetry, blink/jaw triggers,
and the BLE connection state. Press **r** + Enter to recalibrate, **q**
+ Enter to quit.

### Record EEG while producing in Logic

```bash
muse2 record                 # real Muse 2
muse2 record --simulate      # synthetic EEG, no headset
```

Opens a control page at `http://localhost:8770/` with a live trace,
per-channel contact quality, and a big **Start / Stop** button
(`Space` toggles, `M` drops a marker). The headset stays connected the
whole time, so Start/Stop never has to reconnect over Bluetooth. Each take
lands in `recordings/<date_time>_<label>/`:

| file | what |
| --- | --- |
| `eeg.csv` | raw EEG, 256 Hz, µV: `unix_time, package_num, TP9, AF7, AF8, TP10` |
| `motion.csv` | accelerometer + gyro (when the headset streams them) |
| `features.csv` | 4 Hz alpha/beta/theta/focus/calm/blink/jaw (same math as `perform`) |
| `audio.wav` + `audio_sync.csv` | Logic's output via a loopback input, 24-bit, with frame↔time sync rows |
| `midi.csv` | every MIDI message on the selected inputs |
| `markers.csv`, `events.csv`, `session.json` | your marks, connect/drop log, summary |

Every timestamp is host Unix time — the same clock BrainFlow stamps Muse
samples with — so EEG, audio and MIDI line up without any post-hoc sync.

**EEG → MIDI CC (live).** The `record` page has an *EEG → MIDI CC* panel. Tick
*send* and alpha/beta/theta/focus/calm stream as CC 20–24 on channel 1 over the
IAC Driver bus (each self-calibrated to its rolling 5th–95th percentile and
smoothed at 30 Hz). To bind one in Logic: click the plugin knob, press ⌘L,
hit **Learn** on that row (only that CC sweeps for 4 s), press ⌘L again. Set the
track's automation to *Latch* and the performance is written as automation.
Optional: blink → note C3, jaw clench → D3. Every CC sent is logged to `cc.csv`.

**Sonify a recording.**

```bash
muse2 sonify                 # newest recording
muse2 sonify <folder> --speeds 10,40,172 --frames 64
```

Writes `sonify/` inside the recording: `audify_xN.wav` (EEG sped up N×, stereo
L/R hemispheres), a 2048-sample Serum/Vital wavetable built from the cleanest
channel, and `wavetable_drone.wav` (that wavetable played as a scanning drone).
All 16-bit / 44.1 kHz.

**Capturing what Logic plays:** install [BlackHole 2ch](https://existential.audio/blackhole/)
(`brew install blackhole-2ch`), then in Audio MIDI Setup create a
*Multi-Output Device* = your speakers/headphones + BlackHole, and set it as
Logic's output (Logic → Settings → Audio → Output). `muse2 record`
auto-selects BlackHole as its input.

**Logging MIDI / Logic's timeline:** your keyboard shows up automatically.
To also capture Logic's transport position, enable the IAC Driver in Audio
MIDI Setup, then in Logic → Project Settings → Synchronization → MIDI turn on
*Transmit MIDI Clock* and *MMC/MTC* to the IAC bus. Song position and
start/stop then land in `midi.csv` next to the EEG.

### BLE operator tools

```bash
muse2 battery            # one-shot Muse 2 charge level
muse2 bt-reset           # clear stuck BLE connections after a crash
```

## Cloud demo (Railway)

The full pipeline can't run in the cloud — a hosted container has no
Bluetooth radio for the Muse 2 and no speakers for `sounddevice`. But
the pipeline minus those two has a real use: a **shared public demo**
where the visitor's brain is replaced by the synthetic generator,
Lyria still streams real music, and the audio is shipped to every
connected browser as binary WebSocket frames so visitors hear it
through their own speakers.

That's what `--cloud` mode is. One process, one Lyria session, many
browser viewers, all hearing/seeing the same stream:

```bash
muse2 perform --cloud
# Same as: --simulate-eeg --no-browser --no-tui, plus:
#   * binds the server to 0.0.0.0
#   * skips sounddevice; fans Lyria PCM to browsers via WS binary frames
#   * locks Quit + EEG-mode toggle (no single visitor can break it for others)
#   * adds a "click anywhere to enable sound" overlay on connect
#     (browser autoplay policy)
```

### Deploy to Railway

The repo ships a `Dockerfile` + `railway.json`. From the Railway
dashboard:

1. **New Project → Deploy from GitHub repo** → pick
   `AshritVerma/brAIn-music`.
2. Add environment variables under **Variables**:
   - `GEMINI_API_KEY` — required (Lyria + Imagen)
   - `ANTHROPIC_API_KEY` — optional (prompt-guard + seed-evolver)
3. Hit **Deploy**. Railway picks up `railway.json` + `Dockerfile`
   automatically. The container reads `$PORT` from the platform and
   binds the aiohttp server to it.
4. **Settings → Networking → Generate Domain**, open the URL,
   click anywhere to enable sound, type a prompt, hit Start.

The healthcheck path is `/health` and returns a small JSON body
indicating uptime + Lyria activity. Restart-on-failure is configured
with a 5-retry cap.

### Cost reality check

`--cloud` runs Lyria continuously after the first visitor clicks Start,
billed per second. The seed evolver (Imagen 4 Fast) also runs
periodically. Three controls if you care about cost:

- Set `--evolve-chunks 0` in `railway.json`'s `startCommand` to skip
  the evolver entirely (just keep the initial seed image).
- Or stop the Railway service when you're not actively demoing.
- The Lyria session ends only when the orchestrator exits (SIGTERM
  from Railway = graceful shutdown).

### Local container smoke test

```bash
docker build -t brAIn-music .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  brAIn-music
# Open http://localhost:8000/, click to enable sound, type a prompt.
```

## Signal mapping

Every normalized EEG / audio feature is in `[0, 1]` and feeds both the
music and the visuals. Same brain-state value lights up correlated
parameters on each side, which is what makes the audio and visual feel
coupled even though they're generated independently.

| Feature   | Source       | Music (Lyria)        | Visualizer (shader)         |
| --------- | ------------ | -------------------- | --------------------------- |
| alpha     | EEG          | brightness           | blur radius                 |
| beta      | EEG          | density              | contrast + kaleido segments |
| theta     | EEG          | temperature          | tunnel twist + kaleido drift|
| asymmetry | EEG          | bpm tilt             | hue-shift L/R modulator     |
| blink     | EEG (peak)   | (none, future)       | screen flash                |
| jaw       | EEG (peak)   | (none, future)       | radial shockwave            |
| rms       | audio FFT    | (none — audio out)   | zoom-pulse + brightness     |
| centroid  | audio FFT    | (none — audio out)   | hue temperature shift       |
| onset     | audio FFT    | (none — audio out)   | chromatic-aberration kick   |

## Visual regimes

The shader doesn't have a single "look". Every 15–30 s the visualizer
cross-fades between four distinct **regimes**, each a different warp
of the seed texture:

- **calm** — subtle drift, image stays mostly stable
- **tunnel** — log-spiral zoom into infinity
- **kaleidoscope** — angular fold into 6–12 mirror segments
- **ripple** — concentric water-waves emanating from center

A slow autonomous LFO also walks the hue continuously so the colors
are always shifting, even at neutral EEG. Brain features modulate
intensity *within* whichever regime is active.

## Project layout

```
src/muse2_music_lab/
  config.py            # all tunable constants - edit first
  state.py             # AppState dataclass (cross-task shared state)
  main.py              # asyncio orchestrator entry point
  cli.py               # 'muse2' / 'muse2-music' argparse entry point
  perform_tui.py       # rich.Live TUI for `muse2 perform`
  tui.py               # rich.Live TUI for `muse2 run`
  battery.py           # `muse2 battery`
  bt_reset.py          # `muse2 bt-reset`
  simulate.py          # synthetic-EEG generator

  eeg/
    board.py           # BrainFlow wrapper for Muse 2 BLE
    features.py        # band powers, asymmetry, blink, jaw
    smoother.py        # EMA + baseline calibration + tanh normalize
    brainflow_loop.py  # async EEG loop + supervisor (real <-> sim)

  lyria/
    session.py         # Lyria RealTime session + reconnect supervisor
    mapping.py         # EEG features -> Lyria control parameters
    audio_play.py      # sounddevice playback draining Lyria PCM

  audio/
    fft.py             # real-time RMS / centroid / onset features

  music/
    prompt_guard.py    # Claude rewriter for Lyria-filtered prompts

  server/
    app.py             # aiohttp HTTP/WS server + browser action router
    audio_broadcast.py # cloud-mode binary-WS Lyria audio fan-out

  visuals/
    seed_image.py      # Imagen 4 Fast seed generation + cache
    seed_evolver.py    # Claude-driven Imagen regenerations every N chunks

static/
  index.html           # browser UI shell
  style.css            # dark UI styling
  app.js               # WebSocket client + UI controls + seed cross-fade
  visualizer.js        # Three.js + GLSL multi-regime shader
  audio.js             # cloud-mode Web Audio jitter-buffered playback

scripts/
  lyria_smoke.py       # standalone Lyria RealTime API test

Dockerfile             # container image for Railway / any PaaS
railway.json           # Railway build/deploy config (healthcheck etc.)
.dockerignore          # keeps the build context lean
```

## Troubleshooting

- **"No Muse found" / BrainFlow hangs**: make sure the headset is
  charged, powered on (LED breathing pattern), and not held by another
  app (Muse, Mind Monitor, etc.). If the previous run crashed, try
  `muse2 bt-reset`. The `EEG: real / simulated` browser toggle lets
  you start in simulated mode and swap to real once the headset is back.
- **Lyria says nothing arrives / silent audio**: check that
  `GEMINI_API_KEY` is set and has Lyria RealTime access enabled (the
  model is `lyria-realtime-exp`). Run `python scripts/lyria_smoke.py`
  for a focused test.
- **Visualizer is black**: open DevTools, look for `[viz]` log lines.
  A red banner over the canvas means a shader compile error or a
  texture load failure. The fallback radial gradient kicks in if
  `static/seed.png` doesn't load — it's intentionally ugly so you
  can tell the difference.
- **EEG values pegged at 0.50 after recalibrate**: the headset lost
  contact and the smoother baselined against a flat signal. Re-seat
  the band, wait for the alpha/beta bars to wiggle, then recalibrate
  again.
- **Process lingers after Quit**: BrainFlow's underlying C++ BLE
  discovery thread can outlive the Python task by ~10–20 s. The
  Python orchestrator returns cleanly; the OS reaps the BLE thread
  shortly after.

## License

MIT
