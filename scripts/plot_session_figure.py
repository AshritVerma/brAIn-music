import csv, json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
rec = sorted(Path("/Users/ashrit/Code/muse2-music-lab/recordings").glob("*_muse-eeg"))[-1]
rows = list(csv.reader(open(rec/"eeg.csv")))[1:]
x = np.array([[float(v) for v in r[2:6]] for r in rows]).T; fs = 256
t = np.arange(x.shape[1])/fs
meta = json.loads((rec/"session.json").read_text()); ev = meta["events"]
CH = ["TP9","AF7","AF8","TP10"]; COL = ["#2a78d6","#eb6834","#1baf7a","#eda100"]
INK, INK2, GRID, SURF = "#1f1f1e", "#62615c", "#e6e5e0", "#fcfcfb"
def dodge(vals, gap):
    order = np.argsort(vals); out = np.array(vals, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap: out[b] = out[a] + gap
    return out
def bp(sig, lo, hi):
    F = np.fft.rfft(sig - sig.mean()); f = np.fft.rfftfreq(len(sig), 1/fs)
    return np.fft.irfft(np.where((f>=lo)&(f<=hi), F, 0), len(sig))
plt.rcParams.update({"font.family":["Helvetica","Arial Unicode MS","DejaVu Sans"],"font.size":10,"axes.edgecolor":GRID,"axes.labelcolor":INK2,
  "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"axes.facecolor":SURF,"figure.facecolor":SURF})
fig = plt.figure(figsize=(13,9.6)); gs = fig.add_gridspec(3,2, height_ratios=[1.35,1,1], hspace=0.55, wspace=0.18)
fig.text(0.06,0.965,"Muse 2 session — 4-channel EEG, 256 Hz, 6 min", fontsize=15, weight="bold", color=INK)
fig.text(0.06,0.94,"Raw µV traces, band power and detected events (blinks, jaw clenches). Source of all three 'midterm scratch' versions.", fontsize=9.5, color=INK2)
# A: raw traces 10 s window around a blink
t0 = 40.0; w = (t>=t0)&(t<t0+10)
ax = fig.add_subplot(gs[0,:]); off = [300,100,-100,-300]
for i in range(4):
    y = bp(x[i],1,45)[w]
    ax.plot(t[w], y+off[i], color=COL[i], lw=1.1)
    ax.text(t0+10.08, off[i], CH[i], color=INK, va="center", fontsize=10, weight="bold")
for e in ev:
    if t0 <= e["t"] < t0+10:
        ax.axvline(e["t"], color=INK2, lw=0.8, ls=(0,(3,3)))
        ax.text(e["t"], 430, e["kind"], ha="center", fontsize=8.5, color=INK2)
ax.set_xlim(t0, t0+10); ax.set_ylim(-420, 460); ax.set_yticks([]); ax.set_xlabel("time (s)")
ax.set_title("A  Raw signal, 1–45 Hz (10-second window): blinks spike the forehead sensors AF7/AF8", loc="left", fontsize=11, color=INK, weight="bold")
ax.plot([t0+0.15,t0+0.15],[-400,-300], color=INK, lw=2); ax.text(t0+0.25,-350,"100 µV",fontsize=8.5,color=INK2,va="center")
# B: band power over the session (z-scored, 2 s windows)
ax = fig.add_subplot(gs[1,:]); win = 2*fs; n = x.shape[1]//win
bands = [("alpha 8–12 Hz",8,12,COL[0]),("beta 13–30 Hz",13,30,COL[1]),("theta 4–8 Hz",4,8,COL[2])]
tt = (np.arange(n)+0.5)*2
ends = []
for name,lo,hi,c in bands:
    p = np.array([np.mean([np.mean(bp(x[ch,k*win:(k+1)*win],lo,hi)**2) for ch in range(4)]) for k in range(n)])
    z = (np.log10(p)-np.log10(p).mean())/np.log10(p).std(); z = np.convolve(z, np.ones(5)/5, mode="same")
    ax.plot(tt, z, color=c, lw=2, label=name); ends.append(z[-3])
for (name,lo,hi,c), y in zip(bands, dodge(ends, 0.42)):
    ax.text(tt[-1]+3, y, name, color=INK, fontsize=9.5, va="center")
ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9, handlelength=1.6, bbox_to_anchor=(0, 1.02))
ax.axhline(0, color=GRID, lw=1); ax.set_xlim(0, tt[-1]+45); ax.set_ylim(-2.2, 2.6); ax.set_xlabel("time (s)"); ax.set_ylabel("power (z-score)")
ax.set_title("B  Band power across the session — what drives the CC automation (alpha → cutoff, beta → pan, calm → strings)", loc="left", fontsize=11, color=INK, weight="bold")
ax.grid(axis="y", color=GRID, lw=0.6)
# C: PSD per channel
ax = fig.add_subplot(gs[2,0]); cends = []
for i in range(4):
    seg = x[i]-x[i].mean(); k = 4*fs; m = len(seg)//k
    P = np.mean([np.abs(np.fft.rfft(seg[j*k:(j+1)*k]*np.hanning(k)))**2 for j in range(m)],0); f = np.fft.rfftfreq(k,1/fs)
    s = (f>=1)&(f<=40); ax.semilogy(f[s], P[s], color=COL[i], lw=1.6, label=CH[i])
    cends.append(np.log10(P[s][-1]))
for i, y in enumerate(dodge(cends, 0.3)):
    ax.text(40.6, 10**y, CH[i], fontsize=9, color=INK, va="center")
ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=8.5, handlelength=1.4)
ax.axvspan(8,12,color="#2a78d6",alpha=0.07); ax.text(10,ax.get_ylim()[1]*0.5,"alpha",ha="center",fontsize=8.5,color=INK2)
ax.set_xlim(1,46); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (log)"); ax.set_yticks([])
ax.set_title("C  Spectrum per sensor: alpha peak strongest behind the ears", loc="left", fontsize=11, color=INK, weight="bold")
# D: event counts / summary
ax = fig.add_subplot(gs[2,1]); ax.axis("off")
nb = sum(e["kind"]=="blink" for e in ev); nj = sum(e["kind"]=="jaw" for e in ev)
lines = [("Duration", f"{x.shape[1]/fs:.0f} s  ({x.shape[1]:,} samples × 4 sensors)"),("Sample rate","256 Hz"),("Sensors","TP9, AF7, AF8, TP10 (Muse 2 layout)"),
 ("Blinks", f"{nb}  → low thumps in the hits file"),("Jaw clenches", f"{nj}  → bright snaps in the hits file"),
 ("Used for","1 neural automation · 2 audification · 3 EEG wavetable"),("Recording", rec.name)]
ax.set_title("D  Session summary", loc="left", fontsize=11, color=INK, weight="bold")
for k,(a,b) in enumerate(lines):
    ax.text(0.0, 0.88-k*0.13, a, fontsize=10, color=INK2, transform=ax.transAxes)
    ax.text(0.27, 0.88-k*0.13, b, fontsize=10, color=INK, transform=ax.transAxes)
out = Path("/Users/ashrit/Music/Logic/21M/project/Muse capture/muse2_session.png")
fig.savefig(out, dpi=160, bbox_inches="tight"); print(out, nb, nj)
