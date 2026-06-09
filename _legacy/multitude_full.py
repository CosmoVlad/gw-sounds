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

AU = 500
jnp = np

# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))


# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel()
# ax.set_ylabel()

# %%
def Ln(thetaS,phiS,thetaL,phiL):
    
    return jnp.cos(thetaL)*jnp.cos(thetaS) + jnp.sin(thetaL)*jnp.sin(thetaS)*jnp.cos(phiL-phiS)
    
def convert_angle(angle):
    
    return angle + jnp.piecewise(angle, [angle<0.], [2*jnp.pi])

def source_polarization(thetaS, phiS, thetaL, phiL):
    
    return jnp.arctan2(
        jnp.sin(thetaL)*jnp.sin(phiL-phiS), 
        jnp.sin(thetaL)*jnp.cos(phiL-phiS)*jnp.cos(thetaS) - jnp.cos(thetaL)*jnp.sin(thetaS)
    )

def source_inclination(thetaS, phiS, thetaL, phiL):
    
    return jnp.arccos(Ln(thetaS,phiS,thetaL,phiL))

def PhiL(thetaS, phiS, iota, Phi):
    
    return jnp.arctan2(
        jnp.cos(iota)*jnp.sin(thetaS)*jnp.sin(phiS) + jnp.sin(iota)*(
            jnp.sin(Phi)*jnp.cos(phiS) + jnp.cos(Phi)*jnp.sin(phiS)*jnp.cos(thetaS)
        ),
        jnp.cos(iota)*jnp.sin(thetaS)*jnp.cos(phiS) - jnp.sin(iota)*(
            jnp.sin(Phi)*jnp.sin(phiS) - jnp.cos(Phi)*jnp.cos(phiS)*jnp.cos(thetaS)
        )
    )

def ThetaL(thetaS, phiS, iota, Phi):
    
    return jnp.arccos(
        jnp.cos(iota)*jnp.cos(thetaS) - jnp.sin(iota)*jnp.sin(thetaS)*jnp.cos(Phi)
    )
    
## LISA pattern functions in terms of (iota, Phi)
## time is measured in yr

def phi_t(t):
    
    return 2*jnp.pi*t

def expr_cos(t, theta, phi):
    
    return jnp.cos(theta)/2 - jnp.sqrt(3)/2 * jnp.sin(theta)*jnp.cos(phi_t(t) - phi)

def Lz(t, thetaL, phiL):
    
    return expr_cos(t, thetaL, phiL)

def expr_cos_thetaS(t, thetaS, phiS):
    
    return expr_cos(t, thetaS, phiS)


def expr_phiS(t, thetaS, phiS):
    
    return phi_t(t) + jnp.arctan((jnp.sqrt(3)*jnp.cos(thetaS) + jnp.sin(thetaS)*jnp.cos(phi_t(t)-phiS))\
                             /(2*jnp.sin(thetaS)*jnp.sin(phi_t(t)-phiS)))

def nLz(t, thetaS, phiS, thetaL, phiL):
    
    A = jnp.cos(thetaL)*jnp.sin(thetaS)*jnp.sin(phiS) - jnp.cos(thetaS)*jnp.sin(thetaL)*jnp.sin(phiL)
    B = jnp.cos(thetaS)*jnp.sin(thetaL)*jnp.cos(phiL) - jnp.cos(thetaL)*jnp.sin(thetaS)*jnp.cos(phiS)
    
    return jnp.sin(thetaL)*jnp.sin(thetaS)*jnp.sin(phiL-phiS)/2\
                - jnp.sqrt(3)/2 * jnp.cos(phi_t(t)) * A\
                - jnp.sqrt(3)/2 * jnp.sin(phi_t(t)) * B


def polarization_angle(t, thetaS, phiS, Phi):
    
    z1 = jnp.sqrt(3)/2 * jnp.cos(thetaS) * jnp.cos(phi_t(t)-phiS) + 0.5 * jnp.sin(thetaS)
    z2 = jnp.sqrt(3)/2 * jnp.sin(phi_t(t)-phiS)
    
    return jnp.arctan2((-z1*jnp.cos(Phi) - z2*jnp.sin(Phi)),(z1*jnp.sin(Phi) - z2*jnp.cos(Phi)))
    
def shift(phase, dphase):
    
    return phase - dphase

def Fplus(t, thetaS, phiS, iota, Phi, phase_shift=0):
    
    cosThetaS = expr_cos_thetaS(t, thetaS, phiS)
    pol_angle = polarization_angle(t, thetaS, phiS, Phi)
    PhiS = shift(expr_phiS(t, thetaS, phiS),phase_shift)
    #PhiS -= phase_shift
    
    return (1+cosThetaS**2)/2 * jnp.cos(2*PhiS)*jnp.cos(2*pol_angle)\
                        - cosThetaS * jnp.sin(2*PhiS)*jnp.sin(2*pol_angle)

def Fcross(t, thetaS, phiS, iota, Phi, phase_shift=0):
    
    cosThetaS = expr_cos_thetaS(t, thetaS, phiS)
    pol_angle = polarization_angle(t, thetaS, phiS, Phi)
    PhiS = shift(expr_phiS(t, thetaS, phiS), phase_shift)
    #PhiS -= phase_shift
    
    return (1+cosThetaS**2)/2 * jnp.cos(2*PhiS)*jnp.sin(2*pol_angle)\
                        + cosThetaS * jnp.sin(2*PhiS)*jnp.cos(2*pol_angle)


# %%
from scipy.signal.windows import tukey
from scipy.io.wavfile import write

fmin = 1e+2
fmax = 4e+3
lisa_period = 5.


# generate a GW sound of length [s] for equal-mass binaries of masses mm [Msun] at frequencies ff [Hz] 

def gen_sounds(duration,ff,mm, sr=44100, fmin=fmin, fmax=fmax,\
               doppler=False, lisa_period=lisa_period, thetaS=np.pi/2, phiS=0.01):
    
    length = int(sr*duration)
    
    numsignals = len(ff)
    
    eta = 0.25
    Mc = eta**(3./5) * mm * 2
    Mc *= Msun
    
    tc = 5*Mc/256. * (np.pi*Mc*ff)**(-8./3)
    tc /= sr
    
#     print(tc.min())
    
    
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
    if doppler:
        phase += 2*np.pi * 0.1*freq*np.sin(thetaS)*np.cos(2*np.pi*times/lisa_period - phiS)
    
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

name = 'multitude_full'

samplerate = 2*44100
numsignals = 100

name += '_{:d}'.format(numsignals)


amplitude = np.iinfo(np.int16).max

duration = 10

mm = 10**(1 + 3*rng.random(numsignals))
ff = 10**(-4 + 2*rng.random(numsignals))

thetaS = np.arccos(-1 + 2*rng.random(numsignals))
phiS = 2*np.pi*rng.random(numsignals)
iota = np.arccos(-1 + 2*rng.random(numsignals))
phi = 2*np.pi*rng.random(numsignals)

chunks = []
lisa1 = []
lisa2 = []

for index,(m,f,thth,phph,incl,pol) in enumerate(zip(mm,ff,thetaS,phiS,iota,phi)):
    
    if index % 10 == 0:
        print('count: {:d}'.format(index // 10))

    time_curr,chunk = gen_sounds(
        duration, np.array([f]), np.array([m]), 
        sr=samplerate, thetaS=thth, phiS=phph,
    )
    
    chunks.append(chunk)
    
    time_curr,chunk = gen_sounds(
        duration, np.array([f]), np.array([m]), 
        doppler=True, sr=samplerate, thetaS=thth, phiS=phph,
    )
    
    Fp_I = Fplus(time_curr/lisa_period, thth, phph, incl, pol, phase_shift=0.)
    Fc_I = Fcross(time_curr/lisa_period, thth, phph, incl, pol, phase_shift=0.)
    Fp_II = Fplus(time_curr/lisa_period, thth, phph, incl, pol, phase_shift=np.pi/4)
    Fc_II = Fcross(time_curr/lisa_period, thth, phph, incl, pol, phase_shift=np.pi/4)


    lisa_chunk1 = chunk * (Fp_I*(1+np.cos(incl)**2) + Fc_I*2*np.cos(pol))
    lisa_chunk2 = chunk * (Fp_II*(1+np.cos(incl)**2) + Fc_II*2*np.cos(pol))
    
    lisa1.append(lisa_chunk1)
    lisa2.append(lisa_chunk2)
    
signal = np.sum(chunks, axis=0)

lisa_signal1 = np.sum(lisa1, axis=0)
lisa_signal2 = np.sum(lisa2, axis=0)

time = np.concatenate((time_curr, time_curr + time_curr[-1]))
    

if np.abs(signal).max() != 0.:
    signal = amplitude * signal/np.abs(signal).max()
    
if np.abs(lisa_signal1).max() != 0.:
    lisa_signal1 = amplitude * lisa_signal1/np.abs(lisa_signal1).max()
if np.abs(lisa_signal2).max() != 0.:
    lisa_signal2 = amplitude * lisa_signal2/np.abs(lisa_signal2).max()
    
noise_signal1 = lisa_signal1  + np.max(lisa_signal1) * 0.1*rng.normal(size=len(lisa_signal1))
noise_signal2 = lisa_signal2  + np.max(lisa_signal2) * 0.1*rng.normal(size=len(lisa_signal2))

if np.abs(noise_signal1).max() != 0.:
    noise_signal1 = amplitude * noise_signal1/np.abs(noise_signal1).max()
if np.abs(noise_signal2).max() != 0.:
    noise_signal2 = amplitude * noise_signal2/np.abs(noise_signal2).max()
    
pure_data = np.array([signal,signal], dtype=np.int16)
lisa_data = np.array([lisa_signal1,lisa_signal2], dtype=np.int16)
noise_data = np.array([noise_signal1,noise_signal2], dtype=np.int16)

data = np.concatenate((pure_data, lisa_data, noise_data), axis=1)


# %%
# amplitude = np.iinfo(np.int16).max
# data = amplitude * signal/np.abs(signal).max()

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(10,8))

ax.plot(time_curr, pure_data[0], label='GW')

lines = []

for k,stream in enumerate(lisa_data):
    label = 'GW + LISA arm {:d}'.format(k+1)
    line = ax.plot(
        time_curr + time_curr[-1],stream, 
        label=label,
        alpha=0.6
    )
    lines.append(line)
    
for k,(stream, line) in enumerate(zip(noise_data,lines)):
    ax.plot(
        time_curr + 2*time_curr[-1],stream, 
        alpha=0.6,
        c = line[0].get_color()
    )


ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time, s')
ax.set_ylabel('amplitude')

ax.legend()

# fig.tight_layout()
# fig.savefig('{}.jpg'.format(name))

# write('{}.wav'.format(name), samplerate, data.T)

# %%

for k,chunk in enumerate([pure_data, lisa_data, noise_data]):

    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(7,5))

    if k == 0:
        ax.loglog(*better_fft(chunk[0], time_curr, inc=1), label='GW')

    if k > 0:
        for i,stream in enumerate(chunk):
            label = 'GW + LISA arm {:d}'.format(i+1)
            ax.loglog(
                *better_fft(stream, time_curr + k*time_curr[-1], inc=1), 
                label=label,
                alpha=0.6
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

#     fig.tight_layout()
#     fig.savefig('{}_fft_{:d}.jpg'.format(name,k))

# %%
