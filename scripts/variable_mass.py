"""A 'scale' of single sources across a mass range (no LISA response).

Sweeps the component mass over a logspace grid (default 1e4..1e7 Msun) at fixed
GW frequency, sonifies each as its own clip, and concatenates them into one
track -- a ladder of tones illustrating how mass sets the pitch/chirp. Writes
audio/variable_mass.wav and figures/variable_mass{,_fft}.jpg.

Run:  python scripts/variable_mass.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gwsounds.sonify import gen_sounds, better_fft, sci_format, apply_plot_style, wav_write, FMIN, FMAX

SEED = 42
SR = 2 * 44100
NUMSIGNALS = 7
NAME = "variable_mass"

AUDIO = ROOT / "audio"; AUDIO.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
mm = np.logspace(4, 7, NUMSIGNALS)             # component masses [Msun]
ff = np.full_like(mm, 1e-3)                     # fixed GW frequency [Hz]
durations = np.full_like(mm, 3.)               # per-tone length [s]

amplitude = np.iinfo(np.int16).max
chunks, times, ffts = [], [], []
shift = 0.
for f, m, dur in zip(ff, mm, durations):
    time, chunk = gen_sounds(dur, np.array([f]), np.array([m]), sr=SR, rng=rng)
    if np.abs(chunk).max() != 0.:
        chunk = amplitude * chunk / np.abs(chunk).max()
    times.append(time + shift)
    chunks.append(chunk)
    ffts.append(better_fft(chunk, time, inc=1))
    shift += dur

signal = np.concatenate(chunks)
time_full = np.concatenate(times)
data = signal.astype(np.int16)

wav_write(str(AUDIO / f"{NAME}.wav"), SR, data)
print(f"wrote {AUDIO / f'{NAME}.wav'}")

apply_plot_style()
fig, ax = plt.subplots(figsize=(7, 5))
lines = [ax.plot(times[i], chunks[i])[0] for i in range(NUMSIGNALS)]
ax.grid(True, linestyle=":", linewidth=1.)
ax.set_xlabel("time, s"); ax.set_ylabel("amplitude")
fig.tight_layout(); fig.savefig(FIG / f"{NAME}.jpg")

fig, ax = plt.subplots(figsize=(7, 5))
for (x, y), line, m in zip(ffts, lines, mm):
    ax.loglog(x, y, lw=2, c=line.get_color(), label=fr"${sci_format(m)}\ M_\odot$")
ax.grid(True, linestyle=":", linewidth=1.)
ax.set_xlim(FMIN, FMAX); ax.set_ylim(1e-1, 1e9)
ax.set_xlabel("frequency, Hz"); ax.set_ylabel(r"$\left|h_F\right|^2$"); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(FIG / f"{NAME}_fft.jpg")
print(f"wrote figures/{NAME}.jpg, figures/{NAME}_fft.jpg")
