"""Single equal-mass inspiral shifted to the audible band (no LISA response).

The simplest sonification: one source, one chirp sweeping up as it inspirals.
Compare with mono_response.py, which adds LISA's orbital Doppler + antenna
modulation. Writes audio/mono.wav and figures/mono{,_fft}.jpg.

Run:  python scripts/mono.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # make the gwsounds package importable

import numpy as np
import matplotlib
matplotlib.use("Agg")                  # headless: save figures, no display
import matplotlib.pyplot as plt

from gwsounds.sonify import gen_sounds, better_fft, apply_plot_style, wav_write, FMIN, FMAX

SEED = 42
SR = 2 * 44100          # audio sample rate [Hz]
DURATION = 10           # clip length [s]
NAME = "mono"

AUDIO = ROOT / "audio"; AUDIO.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
mm = 10**(1 + 2 * rng.random(1))       # component mass [Msun]
ff = 10**(-3.5 + rng.random(1))        # GW frequency [Hz]

time, signal = gen_sounds(DURATION, ff, mm, sr=SR, rng=rng)

amplitude = np.iinfo(np.int16).max
if np.abs(signal).max() != 0.:
    signal = amplitude * signal / np.abs(signal).max()
data = signal.astype(np.int16)

wav_write(str(AUDIO / f"{NAME}.wav"), SR, data)
print(f"wrote {AUDIO / f'{NAME}.wav'}")

apply_plot_style()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time, data, label="single source")
ax.grid(True, linestyle=":", linewidth=1.)
ax.set_xlabel("time, s"); ax.set_ylabel("amplitude"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / f"{NAME}.jpg")

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(*better_fft(data, time, inc=2), label="single source")
ax.grid(True, linestyle=":", linewidth=1.)
ax.set_xlim(FMIN, FMAX)
ax.set_xlabel("frequency, Hz"); ax.set_ylabel(r"$\left|h_F\right|^2$"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / f"{NAME}_fft.jpg")
print(f"wrote figures/{NAME}.jpg, figures/{NAME}_fft.jpg")
