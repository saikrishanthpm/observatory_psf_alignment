import batoid
import numpy as np

from .utils import utils

class correction:

    def removeTiltWavefrontBrut(WF, wl, tiltX=True, tiltY=True):
        if isinstance(WF, batoid.lattice.Lattice):
            WF = WF.array
            
        rows, cols = WF.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)

        X = np.vstack((x.flatten(), y.flatten())).T
        Z = WF.flatten()

        coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
        a, b = coeffs
        if tiltX and tiltY:
            fitted_plane = a * x + b * y
            removed = np.array([a, b]) * -wl * 1e9
        elif tiltX and not tiltY:
            fitted_plane = a * x
            removed = np.array([a, 0.0]) * -wl * 1e9
        elif not tiltX and tiltY:
            fitted_plane = b * y
            removed = np.array([0.0, b]) * -wl * 1e9

        corrWF = WF - fitted_plane.reshape(rows, cols)
        avg = (np.max(corrWF) + np.min(corrWF)) / 2

        return corrWF - avg, removed

    def removeTiltWavefront(WF, Znk, x, y, removeTilt=True, removePiston=False):
        removed = []
        if removePiston:
            Piston = Znk[1]
            removed.append(Piston)
            Znk[1] = 0.0
        if removeTilt:
            X_tilt, Y_tilt = 2*Znk[2], 2*Znk[3]
            removed.append(X_tilt)
            removed.append(Y_tilt)
            Znk[2], Znk[3] = 0.0, 0.0
    
        N = np.shape(WF)[0]

        W = utils.wavefrontFromZernike(Znk, x, y, WF.mask, N)
        
        return W, removed
