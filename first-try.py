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
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))


# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel()
# ax.set_ylabel()

# %% [markdown]
# ### Generate noise in a frequency band
#
# #### Flat spectrum
#
# Here we generate noise by sampling frequencies from a log-uniform distribution between $10^{-3}\;\mbox{Hz}$ and $1\;\mbox{Hz}$ and phases, from a uniform distribution between 0 and $2\pi$.

# %%
length = 5
samplerate = 44100
numsignals = 1000

rng = np.random.default_rng()

#ff = np.tile(10 + 9990*rng.random(numsignals), (length*samplerate,1)).T
ff = np.tile(10**(1 + 3*rng.random(numsignals)), (length*samplerate,1)).T
psi = np.tile(2*np.pi*rng.random(numsignals), (length*samplerate,1)).T


times = np.linspace(0,length,length*samplerate)
signal = np.sum(np.cos(2*np.pi*ff*times + psi), axis=0)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))

ax.plot(times, signal)


ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time, s')
ax.set_ylabel('amplitude')


# %%
from scipy.io.wavfile import write


fs = 200
fd = fs/length

amplitude = np.iinfo(np.int16).max

def signal_func(fs,times):
    
    return np.sin(2.*np.pi*fs*times + np.pi*fd*times**2)

signal = np.sum([signal_func(fs+i,times)/(i**2+1) for i in np.arange(9)-4], axis=0)


#signal = rng.normal(size=length*samplerate)
data = amplitude * signal/np.abs(signal).max()




#data = amplitude * np.sin(2. * np.pi * fs * t)
# data2 = amplitude * np.sin(2. * np.pi * fs * t)

# data = np.array([data1,data2]).T

write("example.wav", samplerate, data.astype(np.int16))

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))

ax.hist(ff[:,0], bins=np.logspace(1,4,20), density=True)

ax.set_xscale('log')
ax.set_yscale('log')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)




# %%
np.arange(9) - 4

# %%

signal.shape

# %%
plt.plot(times,data)

# %%
