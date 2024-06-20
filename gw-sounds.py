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
# generate a GW sound of length [s] for equal-mass binaries of masses mm [Msun] at frequencies ff [Hz] 

def gen_sounds(duration,ff,mm, sr=44100):
    
    length = int(sr*duration)
    
    numsignals = len(ff)
    
    eta = 0.25
    Mc = eta**(3./5) * mm
    Mc *= Msun
    
    tc = 5*Mc/256. * (np.pi*Mc*ff)**(-8./3)
    
    ff = 100*(ff/1e-4)**(np.log10(40)/2)
    tc /= sr
    
    print(np.sort(tc)[:10])
    
    
    freq = np.tile(ff, (length,1)).T
    tmerge = np.tile(tc, (length,1)).T
    mchirp = np.tile(eta**(3./5) * mm, (length,1)).T
    psi = np.tile(2*np.pi*rng.random(numsignals), (length,1)).T
    A = 1.
    
    
    times = np.linspace(0,duration,length)
    
    cond = tmerge > times
    factor = np.where(
        cond,
        (1-times/tmerge),
        1.
    )

    
    phase = 2*np.pi*freq * (8./5)*tmerge
    phase *=(1 - factor**(5./8))
    
    
    signal = np.sum(mchirp**(5./3)*(freq*factor**(-3./8))**(2./3)*np.cos(phase + psi), axis=0)
    
    amplitude = np.iinfo(np.int16).max
    
    return times, amplitude * signal/np.abs(signal).max()



# %%
rng = np.random.default_rng()

numsignals = 100
samplerate = 2*44100

ff = 10**(-4 + 2*rng.random(numsignals))
mm = 0.5 + rng.random(numsignals)

times,data = gen_sounds(5.3, ff, mm, sr=samplerate)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))

ax.plot(times, data)


ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time, s')
ax.set_ylabel('amplitude')

# %%
from scipy.io.wavfile import write

write("example.wav", samplerate, data.astype(np.int16))

# %%
