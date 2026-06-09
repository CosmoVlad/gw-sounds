"""LISA antenna-pattern (detector response) functions.

Time `t` is measured in LISA orbital periods (i.e. pass t / LISA_PERIOD). The two
output beam-pattern functions are Fplus / Fcross; the rest are the geometry
helpers they are built from. Extracted verbatim from the original notebooks
(the `jnp = np` alias is dropped — plain NumPy here).
"""
import numpy as np


def Ln(thetaS, phiS, thetaL, phiL):
    return np.cos(thetaL) * np.cos(thetaS) + \
        np.sin(thetaL) * np.sin(thetaS) * np.cos(phiL - phiS)


def convert_angle(angle):
    return angle + np.piecewise(angle, [angle < 0.], [2 * np.pi])


def source_polarization(thetaS, phiS, thetaL, phiL):
    return np.arctan2(
        np.sin(thetaL) * np.sin(phiL - phiS),
        np.sin(thetaL) * np.cos(phiL - phiS) * np.cos(thetaS) - np.cos(thetaL) * np.sin(thetaS),
    )


def source_inclination(thetaS, phiS, thetaL, phiL):
    return np.arccos(Ln(thetaS, phiS, thetaL, phiL))


def PhiL(thetaS, phiS, iota, Phi):
    return np.arctan2(
        np.cos(iota) * np.sin(thetaS) * np.sin(phiS) + np.sin(iota) * (
            np.sin(Phi) * np.cos(phiS) + np.cos(Phi) * np.sin(phiS) * np.cos(thetaS)
        ),
        np.cos(iota) * np.sin(thetaS) * np.cos(phiS) - np.sin(iota) * (
            np.sin(Phi) * np.sin(phiS) - np.cos(Phi) * np.cos(phiS) * np.cos(thetaS)
        ),
    )


def ThetaL(thetaS, phiS, iota, Phi):
    return np.arccos(
        np.cos(iota) * np.cos(thetaS) - np.sin(iota) * np.sin(thetaS) * np.cos(Phi)
    )


# --- LISA pattern functions in terms of (iota, Phi); time in LISA periods -----

def phi_t(t):
    return 2 * np.pi * t


def expr_cos(t, theta, phi):
    return np.cos(theta) / 2 - np.sqrt(3) / 2 * np.sin(theta) * np.cos(phi_t(t) - phi)


def Lz(t, thetaL, phiL):
    return expr_cos(t, thetaL, phiL)


def expr_cos_thetaS(t, thetaS, phiS):
    return expr_cos(t, thetaS, phiS)


def expr_phiS(t, thetaS, phiS):
    return phi_t(t) + np.arctan(
        (np.sqrt(3) * np.cos(thetaS) + np.sin(thetaS) * np.cos(phi_t(t) - phiS))
        / (2 * np.sin(thetaS) * np.sin(phi_t(t) - phiS))
    )


def nLz(t, thetaS, phiS, thetaL, phiL):
    A = np.cos(thetaL) * np.sin(thetaS) * np.sin(phiS) - np.cos(thetaS) * np.sin(thetaL) * np.sin(phiL)
    B = np.cos(thetaS) * np.sin(thetaL) * np.cos(phiL) - np.cos(thetaL) * np.sin(thetaS) * np.cos(phiS)
    return np.sin(thetaL) * np.sin(thetaS) * np.sin(phiL - phiS) / 2 \
        - np.sqrt(3) / 2 * np.cos(phi_t(t)) * A \
        - np.sqrt(3) / 2 * np.sin(phi_t(t)) * B


def polarization_angle(t, thetaS, phiS, Phi):
    z1 = np.sqrt(3) / 2 * np.cos(thetaS) * np.cos(phi_t(t) - phiS) + 0.5 * np.sin(thetaS)
    z2 = np.sqrt(3) / 2 * np.sin(phi_t(t) - phiS)
    return np.arctan2((-z1 * np.cos(Phi) - z2 * np.sin(Phi)), (z1 * np.sin(Phi) - z2 * np.cos(Phi)))


def shift(phase, dphase):
    return phase - dphase


def Fplus(t, thetaS, phiS, iota, Phi, phase_shift=0):
    cosThetaS = expr_cos_thetaS(t, thetaS, phiS)
    pol_angle = polarization_angle(t, thetaS, phiS, Phi)
    PhiS = shift(expr_phiS(t, thetaS, phiS), phase_shift)
    return (1 + cosThetaS**2) / 2 * np.cos(2 * PhiS) * np.cos(2 * pol_angle) \
        - cosThetaS * np.sin(2 * PhiS) * np.sin(2 * pol_angle)


def Fcross(t, thetaS, phiS, iota, Phi, phase_shift=0):
    cosThetaS = expr_cos_thetaS(t, thetaS, phiS)
    pol_angle = polarization_angle(t, thetaS, phiS, Phi)
    PhiS = shift(expr_phiS(t, thetaS, phiS), phase_shift)
    return (1 + cosThetaS**2) / 2 * np.cos(2 * PhiS) * np.sin(2 * pol_angle) \
        + cosThetaS * np.sin(2 * PhiS) * np.cos(2 * pol_angle)
