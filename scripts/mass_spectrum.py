"""Many overlapping sources -- the 'cosmic confusion' demo, in three stages.

Draws N binaries with random masses, frequencies, and sky positions, then builds:
  1. pure     : the bare summed GW signal (mono),
  2. lisa     : same sources passed through the LISA antenna response (stereo,
                two beam-pattern phases), and
  3. noise    : the LISA-response signal plus 10% detector noise (stereo).
Going pure -> lisa -> noise is the 'what LISA actually hears' progression.
Writes audio/mass_spectrum_{pure,lisa,noise}.wav and a comparison figure.

Run:  python scripts/mass_spectrum.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gwsounds.sonify import gen_sounds, better_fft, apply_plot_style, wav_write, LISA_PERIOD, FMIN, FMAX
from gwsounds.response import Fplus, Fcross

SEED = 42
SR = 2 * 44100
DURATION = 10
NUMSIGNALS = 100
NOISE_FRAC = 0.1
NAME = "mass_spectrum"

AUDIO = ROOT / "audio"; AUDIO.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
mm = 10**(1 + 3 * rng.random(NUMSIGNALS))          # component masses [Msun]
ff = 10**(-4 + 2 * rng.random(NUMSIGNALS))         # GW frequencies [Hz]
thetaS = np.arccos(-1 + 2 * rng.random(NUMSIGNALS))
phiS = 2 * np.pi * rng.random(NUMSIGNALS)
iota = np.arccos(-1 + 2 * rng.random(NUMSIGNALS))
phi = 2 * np.pi * rng.random(NUMSIGNALS)

pure_chunks, arm1_chunks, arm2_chunks = [], [], []
for i, (m, f, thth, phph, incl, pol) in enumerate(zip(mm, ff, thetaS, phiS, iota, phi)):
    if i % 10 == 0:
        print(f"source {i}/{NUMSIGNALS}")
    _, bare = gen_sounds(DURATION, np.array([f]), np.array([m]),
                         sr=SR, thetaS=thth, phiS=phph, rng=rng)
    pure_chunks.append(bare)

    time, chunk = gen_sounds(DURATION, np.array([f]), np.array([m]),
                             sr=SR, doppler=True, thetaS=thth, phiS=phph, rng=rng)
    t_orbit = time / LISA_PERIOD
    Fp_I = Fplus(t_orbit, thth, phph, incl, pol, phase_shift=0.)
    Fc_I = Fcross(t_orbit, thth, phph, incl, pol, phase_shift=0.)
    Fp_II = Fplus(t_orbit, thth, phph, incl, pol, phase_shift=np.pi / 4)
    Fc_II = Fcross(t_orbit, thth, phph, incl, pol, phase_shift=np.pi / 4)
    arm1_chunks.append(chunk * (Fp_I * (1 + np.cos(incl)**2) + Fc_I * 2 * np.cos(pol)))
    arm2_chunks.append(chunk * (Fp_II * (1 + np.cos(incl)**2) + Fc_II * 2 * np.cos(pol)))

pure = np.sum(pure_chunks, axis=0)
arm1 = np.sum(arm1_chunks, axis=0)
arm2 = np.sum(arm2_chunks, axis=0)
amplitude = np.iinfo(np.int16).max


def normalize(x):
    return amplitude * x / np.abs(x).max() if np.abs(x).max() != 0. else x


pure_n = normalize(pure)
arm1_n, arm2_n = normalize(arm1), normalize(arm2)
noise1 = normalize(arm1_n + NOISE_FRAC * np.max(arm1_n) * rng.normal(size=len(arm1_n)))
noise2 = normalize(arm2_n + NOISE_FRAC * np.max(arm2_n) * rng.normal(size=len(arm2_n)))

streams = {
    "pure": np.array([pure_n, pure_n]).T.astype(np.int16),    # mono dup -> stereo file
    "lisa": np.array([arm1_n, arm2_n]).T.astype(np.int16),
    "noise": np.array([noise1, noise2]).T.astype(np.int16),
}
for stage, data in streams.items():
    wav_write(str(AUDIO / f"{NAME}_{stage}.wav"), SR, data)
    print(f"wrote {AUDIO / f'{NAME}_{stage}.wav'}")

apply_plot_style()
fig, ax = plt.subplots(figsize=(8, 5))
for stage, label in (("pure", "pure GW"), ("lisa", "+ LISA response"), ("noise", "+ detector noise")):
    mono = streams[stage][:, 0]
    ax.loglog(*better_fft(mono, time, inc=1), alpha=0.7, label=label)
ax.grid(True, linestyle=":", linewidth=1.)
ax.set_xlim(FMIN, FMAX); ax.set_ylim(1e-1, 1e9)
ax.set_xlabel("frequency, Hz"); ax.set_ylabel(r"$\left|h_F\right|^2$"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / f"{NAME}_fft.jpg")
print(f"wrote figures/{NAME}_fft.jpg")
