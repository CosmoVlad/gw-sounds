"""Single source WITH the LISA response.

Adds two effects from LISA's yearly orbit on top of the bare chirp:
  - Doppler phase modulation (gen_sounds(doppler=True)), and
  - antenna-pattern amplitude modulation via Fplus/Fcross.
Two beam-pattern phases (0 and pi/4) give two channels -> stereo, so you can
hear the source swell and pan over the LISA year. Writes audio/mono_response.wav
(stereo) and figures/mono_response.jpg.

Run:  python scripts/mono_response.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gwsounds.sonify import gen_sounds, apply_plot_style, wav_write, LISA_PERIOD
from gwsounds.response import Fplus, Fcross

SEED = 42
SR = 2 * 44100
DURATION = 10
NAME = "mono_response"

AUDIO = ROOT / "audio"; AUDIO.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
m = 10**(1 + 2 * rng.random())                 # component mass [Msun]
f = 10**(-3.5 + rng.random())                  # GW frequency [Hz]
thetaS = np.arccos(-1 + 2 * rng.random())      # sky position
phiS = 2 * np.pi * rng.random()
iota = np.arccos(-1 + 2 * rng.random())        # inclination
phi = 2 * np.pi * rng.random()                 # polarization phase

time, chunk = gen_sounds(DURATION, np.array([f]), np.array([m]),
                         sr=SR, doppler=True, thetaS=thetaS, phiS=phiS, rng=rng)

# Two beam-pattern channels (phase 0 and pi/4), as the two LISA TDI arms.
t_orbit = time / LISA_PERIOD
arms = []
for phase_shift in (0., np.pi / 4):
    Fp = Fplus(t_orbit, thetaS, phiS, iota, phi, phase_shift=phase_shift)
    Fc = Fcross(t_orbit, thetaS, phiS, iota, phi, phase_shift=phase_shift)
    arms.append(chunk * (Fp * (1 + np.cos(iota)**2) + Fc * 2 * np.cos(phi)))

amplitude = np.iinfo(np.int16).max
stereo = []
for arm in arms:
    if np.abs(arm).max() != 0.:
        arm = amplitude * arm / np.abs(arm).max()
    stereo.append(arm.astype(np.int16))
data = np.array(stereo).T                       # shape (n_samples, 2)

wav_write(str(AUDIO / f"{NAME}.wav"), SR, data)
print(f"wrote {AUDIO / f'{NAME}.wav'} (stereo)")

apply_plot_style()
fig, ax = plt.subplots(figsize=(8, 5))
for k, arm in enumerate(stereo):
    ax.plot(time, arm, alpha=0.7, label=f"LISA arm {k + 1}")
ax.grid(True, linestyle=":", linewidth=1.)
ax.set_xlabel("time, s"); ax.set_ylabel("amplitude"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / f"{NAME}.jpg")
print(f"wrote figures/{NAME}.jpg")
