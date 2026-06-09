# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt


MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rcdefaults()

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %matplotlib inline

# %%
year = 3600*24*365
c = 3e+8
G = 6.67e-11

Msun = 2e+30
Msun *= G/c**3

# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))


# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel()
# ax.set_ylabel()

# %% [markdown]
# ### Shifting GW signals to an audible range from $10^2\;\mbox{Hz}$ to $10^4\;\mbox{Hz}$ 
#
# Consider LISA GW signals between $10^{-4}\;\mbox{Hz}$ and $10^{-2}\;\mbox{Hz}$ and shift them to an audible range:
# $$
# \overline{f} = 10^6 f_{\rm GW}\,.
# $$
#
# Also, the sampling rate in LISA is $1\;\mbox{s}^{-1}$ (1 datapoint per second). Let us assume a sampling rate $s_r=44,100\;\mbox{Hz}$ for the audible counterparts. Then, all the times are reduced proportionally, so that the number of sampled datapoints per GW signal remains the same. For example, the observation time $T_{\rm obs}=4\;\mbox{yr}$ becomes $\overline{T}=T_{\rm obs}/s_r=2,860\;\mbox{s}\approx 47\;\mbox{min}$, the duration of an audiotrack that represents the full extend of LISA observations. 
#
# Coalescence times are similarly reduced, so that the same fraction of the audible band is swept during an observation time:
# $$
# \overline{t} = t_{\rm c}/s_r\,,
# $$
# which means that time derivatives of the GW frequency increase:
# $$
# \dot{\overline{f}} = \dot{f}s_r\,, \quad \ddot{\overline{f}} = \ddot{f}s_r^2\,, \quad \ldots\,.
# $$

# %%
from scipy.signal.windows import tukey
from scipy.io.wavfile import write

fmin = 1e+2
fmax = 4e+3


# generate a GW sound of length [s] for equal-mass binaries of masses mm [Msun] at frequencies ff [Hz] 

def gen_sounds(duration,ff,mm, sr=44100, fmin=fmin, fmax=fmax):
    
    length = int(sr*duration)
    
    numsignals = len(ff)
    
    eta = 0.25
    Mc = eta**(3./5) * mm * 2
    Mc *= Msun
    
    tc = 5*Mc/256. * (np.pi*Mc*ff)**(-8./3)
    tc /= sr
    
    print(np.sort(tc)[:10])
    
    
    freq = np.tile(ff, (length,1)).T
    tmerge = np.tile(tc, (length,1)).T
    mchirp = np.tile(eta**(3./5) * mm, (length,1)).T
    psi = np.tile(2*np.pi*rng.random(numsignals), (length,1)).T
    
    
    times = np.linspace(0,duration,length)
    
    cond = tmerge > times
    factor = np.where(
        cond,
        (1-times/tmerge),
        1.
    )
    
    cond_freq = np.logical_and(
        freq*factor**(-3./8) < 1e-2, 
        freq*factor**(-3./8) < 4400/(mchirp/(2*eta**(3./5)))
    )
    cond = np.logical_and(cond, cond_freq)
    
    freq = fmin*(freq/1e-4)**(np.log10(fmax/fmin)/2)

    phase = 2*np.pi*freq * (8./5)*tmerge
    phase *=(1 - factor**(5./8))
    
    ampl = np.where(
        cond,
        mchirp**(5./3)*(freq*factor**(-3./8))**(2./3),
        0.
    )
    
    
    return times, np.sum(ampl*np.cos(phase + psi), axis=0)


def zero_pad(data, inc=0):  # inc defines whether we go for the nearest power of two or more
    """
    This function takes in a vector and zero pads it so it is a power of two.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N)) + inc
    return np.pad(data,(0,int((2**pow_2)-N-1)),'constant')


def better_fft(signal, time, inc=0, tukey_frac=0.05):
    """
    This function takes in a signal and the respective times, 
    multiplies it by a window function, pads the result with zeros,
    and perform an FFT. The output is two arrays: frequencies and the power spectrum.
    """
    
    dt = time[1]-time[0]
    N = len(signal)
    window = tukey(N, tukey_frac)
    signal_tapered = signal*window
    signal_padded = zero_pad(signal_tapered, inc)
    
    freqs = np.fft.rfftfreq(len(signal_padded), dt)
    hf = dt*np.fft.rfft(signal_padded)
    
    return freqs, np.abs(hf)**2

def sci_format(num,digits=0):
    
    b = np.floor(np.log10(num))
    a = num/10**b
    
    if a == 1.:
        return '10^{:d}'.format(int(b))
    
    return ('{{:.{:d}f}}'.format(int(digits)) + '\\times 10^{:d}').format(a,int(b))



# %%
rng = np.random.default_rng()

name = 'variable_mass'

numsignals = 7
samplerate = 2*44100

mm = np.logspace(4,7,numsignals)
ff = np.full_like(mm, 1e-3)
duration = np.full_like(mm, 3.)

chunks = []
times = []
ffts = []
shift = 0.

amplitude = np.iinfo(np.int16).max

for f,m,dur in zip(ff,mm,duration):
    f = np.array([f])
    m = np.array([m])
    time,chunk = gen_sounds(dur, f, m, sr=samplerate)
    
    if np.abs(chunk).max() != 0.:
        chunk = amplitude * chunk/np.abs(chunk).max()
    
    times.append(time + shift)
    chunks.append(chunk)
    ffts.append(better_fft(chunk,time,inc=1))
    
    shift += dur

signal = np.concatenate(chunks)
time = np.concatenate(times)

#data = amplitude * signal/np.abs(signal).max()
data = signal

# %%
# amplitude = np.iinfo(np.int16).max
# data = amplitude * signal/np.abs(signal).max()

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(7,5))

p = []

for i in range(numsignals):
    line = ax.plot(times[i], chunks[i])
    p.append(line)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time, s')
ax.set_ylabel('amplitude')

fig.tight_layout()
fig.savefig('{}.jpg'.format(name))

write('{}.wav'.format(name), samplerate, data.astype(np.int16))

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(7,5))

for (x,y),line,m in zip(ffts[:6],p[:6],mm[:6]):
    
    label = '$' + sci_format(m) + ' M_\\odot$'
    
    ax.loglog(
        x, y, 
        lw=2, c=line[0].get_color(), label=label
    )

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlim(fmin,fmax)
ax.set_ylim(1e-1,1e+9)

ax.set_xlabel('frequenzy, Hz')
ax.set_ylabel('$\\left|h_F\\right|^2$')

ax.legend()

fig.tight_layout()
fig.savefig('{}_fft.jpg'.format(name))

# %%
