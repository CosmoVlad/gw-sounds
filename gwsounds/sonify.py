"""Core GW sonification: map the LISA band to audio and synthesize equal-mass
inspiral chirps, plus the FFT / formatting helpers shared by every script.

The mapping (see README): GW frequencies 1e-4..1e-2 Hz are stretched to an
audible fmin..fmax, and LISA's 1 s cadence is replayed at sr = 44100 Hz, so a
multi-year observation compresses into a short audio clip. A frequency drift
f_dot scales with the sample rate, so inspiral chirps sweep audibly.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.io.wavfile import write as wav_write   # re-exported for scripts

# --- physical constants (geometrized units: Msun carries the G/c^3 factor) ---
YEAR = 3600 * 24 * 365
C = 3e8
G = 6.67e-11
MSUN = 2e30 * G / C**3

# --- audio band + LISA timescale --------------------------------------------
FMIN = 1e2            # bottom of the audible mapping [Hz]
FMAX = 4e3            # top of the audible mapping [Hz]
LISA_PERIOD = 5.0     # audio seconds per LISA orbital year (Doppler/antenna rate)


def apply_plot_style(usetex=True, medium=18, bigger=22):
    """Serif/Times matplotlib rc setup matching the original notebooks.

    Pass usetex=False if a LaTeX install is not available (labels render with
    mathtext instead).
    """
    plt.rcdefaults()
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    plt.rc('text', usetex=usetex)
    plt.rc('font', size=bigger)
    plt.rc('axes', titlesize=bigger)
    plt.rc('axes', labelsize=bigger)
    plt.rc('xtick', labelsize=bigger)
    plt.rc('ytick', labelsize=bigger)
    plt.rc('legend', fontsize=medium)
    plt.rc('figure', titlesize=bigger)


def gen_sounds(duration, ff, mm, sr=44100, fmin=FMIN, fmax=FMAX,
               doppler=False, lisa_period=LISA_PERIOD,
               thetaS=np.pi / 2, phiS=0.01, rng=None):
    """Summed audio waveform for a set of equal-mass inspiralling binaries.

    duration : audio length [s]
    ff       : GW frequencies [Hz]              (1-D array)
    mm       : per-component mass [Msun]        (1-D array; binary total = 2*mm)
    doppler  : add LISA's orbital Doppler phase modulation (needs thetaS, phiS)
    rng      : np.random.Generator for the random initial phases (default: fresh)

    Returns (times, signal): the time grid [s] and the summed strain (arb. units).
    """
    if rng is None:
        rng = np.random.default_rng()

    length = int(sr * duration)
    numsignals = len(ff)

    eta = 0.25
    Mc = eta**(3. / 5) * mm * 2 * MSUN
    tc = 5 * Mc / 256. * (np.pi * Mc * ff)**(-8. / 3) / sr   # coalescence time [audio s]

    freq = np.tile(ff, (length, 1)).T
    tmerge = np.tile(tc, (length, 1)).T
    mchirp = np.tile(eta**(3. / 5) * mm, (length, 1)).T
    psi = np.tile(2 * np.pi * rng.random(numsignals), (length, 1)).T

    times = np.linspace(0, duration, length)

    cond = tmerge > times
    factor = np.where(cond, 1 - times / tmerge, 1.)

    # only emit while still in band and below the chirp-mass-dependent ceiling
    cond_freq = np.logical_and(
        freq * factor**(-3. / 8) < 1e-2,
        freq * factor**(-3. / 8) < 4400 / (mchirp / (2 * eta**(3. / 5))),
    )
    cond = np.logical_and(cond, cond_freq)

    freq = fmin * (freq / 1e-4)**(np.log10(fmax / fmin) / 2)   # GW band -> audio band

    phase = 2 * np.pi * freq * (8. / 5) * tmerge
    phase *= (1 - factor**(5. / 8))
    if doppler:
        phase += 2 * np.pi * 0.1 * freq * np.sin(thetaS) * \
            np.cos(2 * np.pi * times / lisa_period - phiS)

    ampl = np.where(cond, mchirp**(5. / 3) * (freq * factor**(-3. / 8))**(2. / 3), 0.)

    return times, np.sum(ampl * np.cos(phase + psi), axis=0)


def zero_pad(data, inc=0):
    """Zero-pad a vector up to the next power of two (inc adds extra octaves)."""
    N = len(data)
    pow_2 = np.ceil(np.log2(N)) + inc
    return np.pad(data, (0, int((2**pow_2) - N - 1)), 'constant')


def better_fft(signal, time, inc=0, tukey_frac=0.05):
    """Tukey-window, zero-pad, and FFT a signal.

    Returns (freqs, power) = the rfft frequencies and |h_f|^2.
    """
    dt = time[1] - time[0]
    N = len(signal)
    window = tukey(N, tukey_frac)
    signal_padded = zero_pad(signal * window, inc)

    freqs = np.fft.rfftfreq(len(signal_padded), dt)
    hf = dt * np.fft.rfft(signal_padded)
    return freqs, np.abs(hf)**2


def sci_format(num, digits=0):
    """LaTeX 'a x 10^b' string for axis labels (returns '10^b' when a == 1)."""
    b = np.floor(np.log10(num))
    a = num / 10**b
    if a == 1.:
        return '10^{:d}'.format(int(b))
    return ('{{:.{:d}f}}'.format(int(digits)) + '\\times 10^{:d}').format(a, int(b))
