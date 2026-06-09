# gw-sounds

Sonification of LISA gravitational-wave signals: map the milli-hertz GW band to
audio, replay LISA's slow cadence at 44.1 kHz, and listen to inspiral chirps,
the LISA antenna response, and the "cosmic confusion" of many overlapping
Galactic sources.

## How the mapping works

GW frequencies in `1e-4 .. 1e-2 Hz` are stretched to an audible `fmin .. fmax`
(`1e2 .. 4e3 Hz`), and LISA's 1 s sampling is replayed at `sr = 44100 Hz`, so a
multi-year observation compresses into a clip of seconds. Frequency drift scales
with the sample rate, so inspirals sweep audibly. See the docstring of
`gwsounds/sonify.py:gen_sounds` for the details.

## Layout

```
gwsounds/            installable-style library (the shared core)
  sonify.py          gen_sounds(...) synthesizer + FFT/format helpers + constants
  response.py        LISA antenna pattern (Fplus / Fcross + geometry helpers)
scripts/             one runnable experiment each (plain Python, no notebooks)
  mono.py            single source, no response          -> mono.wav
  mono_response.py   single source + Doppler + antenna   -> mono_response.wav (stereo)
  mass_spectrum.py   many sources, pure/LISA/noise stages -> mass_spectrum_{pure,lisa,noise}.wav
  variable_mass.py   a mass-swept ladder of tones         -> variable_mass.wav
audio/               generated .wav clips   (gitignored)
figures/             generated .jpg figures (gitignored)
_legacy/             original jupytext notebooks, pre-refactor (kept for reference)
```

## Running

```bash
python scripts/mono.py
python scripts/mass_spectrum.py
```

Each script seeds its RNG (`SEED = 42`) so clips are reproducible; edit the
constants at the top to taste. Outputs land in `audio/` and `figures/`.

Requires `numpy`, `scipy`, `matplotlib`. Figure labels use LaTeX
(`apply_plot_style(usetex=True)`); pass `usetex=False` if LaTeX is unavailable.

## History

This project was created for the author's PhD thesis defense at Johns Hopkins
University on June 24, 2024, to sonify LISA gravitational-wave signals for the
talk. The original exploratory code lived in jupytext (`py:percent`) notebooks;
on 2026-06-08 it was refactored into the library + scripts layout above, with
the shared `gen_sounds` / FFT helpers and the antenna-response functions
(previously copy-pasted across six files) collected once in `gwsounds/`. The
pre-refactor sources are preserved under `_legacy/`.

## How to cite

If you use this project, please cite the thesis it was built for:

```bibtex
@phdthesis{strokov2024openmic,
  author = {Strokov, Vladimir},
  title  = {Open Mic with {LISA}: Intermediate-Mass Black Holes and Double White Dwarfs},
  school = {Johns Hopkins University},
  year   = {2024},
  month  = jun,
  type   = {PhD thesis},
  url    = {https://jscholarship.library.jhu.edu/items/c5a09349-fc9f-4b4f-9307-bb5023029dc9},
}
```
