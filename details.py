import batoid
import numpy as np
import galsim
import re
from sympy import symbols, trigsimp

from .utils import utils
from .ray import ray
from .correction import correction

class details:
    """Abstract base class defining different computations of the optical system.
    """
    def __init__(self, 
                 optic, fieldX=0, fieldY=0, wavelength=633e-9
                ):
        """
        Parameters
        ----------
        optic : batoid.Optic
            Optical system as a single optic or a compound optic.
        fieldX : float
            Field over X (in degrees). Default is ``0``.
        fieldY : float
            Field over Y (in degrees). Default is ``0``.
        wavelength : float
            Wavelength for this ray (in meters). Default is ``633e-9``.

        """
        self.optic = optic
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.wavelength = wavelength

    def opticalAxis(self):
        opt, wl = self.optic, self.wavelength
        rv = ray(opt, fieldX=0, fieldY=0, wavelength=wl, nbOfRays=1).fromPupilSingle([0, 0])
        opt.trace(rv)
        return rv.toCoordSys(batoid.globalCoordSys)

    def angleImgSpace(self):
        """The function computes the angle of the system in image space.

        Parameters
        ----------
        
        Returns
        -------
        theta_p : float
             Angle theta' in image space. The object is defined as infinity.
        """
        opt, wl = self.optic, self.wavelength
        D = opt.pupilSize
        n = opt.items[-1].inMedium.getN(wl)
        ray = batoid.RayVector.fromStop(x=0, y=D/2,
                                     optic=opt,
                                     wavelength=wl,
                                     dirCos=utils.zemaxToDirCos(np.deg2rad(0), np.deg2rad(0)),
                                     flux=1.)

        opt.trace(ray, reverse=False)
        return np.arccos(ray.vz[0] * n)

    def paraxialWorkingFNumber(self):
        """The function computes the F number of the system in image space.

        Parameters
        ----------
        
        Returns
        -------
        f/# : float
             Paraxial working F number as being 1 / (2 * NA)
        """
        NA = self.NA()
        return 1 / (2 * NA)

    def NA(self):
        """The function computes the numerical aperture of the system in image space.

        Parameters
        ----------
        
        Returns
        -------
        NA : float
             Numerical aperture as being n' * sin(theta'). The object is defined as infinity.
        """
        n = self.optic.items[-1].inMedium.getN(self.wavelength)
        
        return n * np.sin(self.angleImgSpace())

    def FNumber(self):
        """The function computes the F number of the system.

        Parameters
        ----------
        
        Returns
        -------
        f/# : float
             F number as being EFFL/Dep.
        """
        D = self.optic.pupilSize
        EFFL = self.EFFL()
        return np.abs(EFFL) / D
    
    def angleImgSpaceFrom(self, x, y, offFieldX, offFieldY):
        """The function computes the angle of the system in image space from a specific point
           and from a specific off-axis angle.

        Parameters
        ----------
        x : float
             X position in the pupil
        y : float
             Y position in the pupil
        offFieldX : float
             Off-axis angle in degree along X.
        offFieldY : float
             Off-axis angle in degree along Y.
        
        Returns
        -------
        theta_p : float
             Angle theta' in image space.
        """
        opt, wl = self.optic, self.wavelength
        D = opt.pupilSize
        n = opt.items[-1].inMedium.getN(wl)
        ray = batoid.RayVector.fromStop(x=x, y=y,
                                     optic=opt,
                                     wavelength=wl,
                                     dirCos=utils.zemaxToDirCos(np.deg2rad(offFieldX), np.deg2rad(offFieldY)),
                                     flux=1.)

        opt.trace(ray, reverse=False)
        return np.arccos(ray.vz[0] * n)

    def workingNA(self, offFieldX=0, offFieldY=0):
        """The function computes the 'working' numerical aperture of the system in image space.
           The useful image space angle is the difference between the one on the edge of the pupil
           angle and the off-axis angle T0.
           The useful NA defined as 'working' NA is the standard deviation of left/right/top/bottom.
           This is how it is defined in ZEMAX.

        Parameters
        ----------
        offFieldX : float
             Off-axis angle in degree along X. Default is 0.
        offFieldY : float
             Off-axis angle in degree along Y. Default is 0.
        
        Returns
        -------
        NAw : float
             Numerical aperture as being n' * sin(theta'-theta'_off). The object is defined as infinity.
        """
        D = self.optic.pupilSize
        n = self.optic.items[-1].inMedium.getN(self.wavelength)
        
        T0x = self.angleImgSpaceFrom(0, 0, offFieldX, 0)
        T0y = self.angleImgSpaceFrom(0, 0, 0, offFieldY)
        
        if (offFieldX == 0) & (offFieldY == 0):
            NA = self.NA()
            return NA
        elif (offFieldX == 0) & (offFieldY != 0):
            NA1 = n * np.sin(self.angleImgSpaceFrom(0, D/2, 0, offFieldY) - T0y)
            NA2 = n * np.sin(self.angleImgSpaceFrom(0, -D/2, 0, offFieldY) - T0y)
            return np.sqrt(np.mean(np.array([NA1, NA2])**2))
        elif (offFieldX != 0) & (offFieldY == 0):
            NA1 = n * np.sin(self.angleImgSpaceFrom(D/2, 0, offFieldX, offFieldY) - T0x)
            NA2 = n * np.sin(self.angleImgSpaceFrom(-D/2, 0, offFieldX, offFieldY) - T0x)
            return np.sqrt(np.mean(np.array([NA1, NA2])**2))
        else:
            NA1 = n * np.sin(self.angleImgSpaceFrom(0, D/2, 0, offFieldY) - T0y)
            NA2 = n * np.sin(self.angleImgSpaceFrom(0, -D/2, 0, offFieldY) - T0y)
            NA3 = n * np.sin(self.angleImgSpaceFrom(D/2, 0, offFieldX, 0) - T0x)
            NA4 = n * np.sin(self.angleImgSpaceFrom(-D/2, 0, offFieldX, 0) - T0x)
            return np.sqrt(np.mean(np.array([NA1, NA2, NA3, NA4])**2))

    def workingFNumber(self, offFieldX=0, offFieldY=0):
        """The function computes the working F number of the system.

        Parameters
        ----------
        offFieldX : float
             Off-axis angle in degree along X. Default is 0.
        offFieldY : float
             Off-axis angle in degree along Y. Default is 0.
        
        Returns
        -------
        f/#w : float
             Working F number as being 1 / (2 * NAw)
        """
        return 1 / (2 * self.workingNA(offFieldX, offFieldY))

    def airyRadius(self, offFieldX=0, offFieldY=0):
        """The function computes the Airy Radius (diffraction limit).

        Parameters
        ----------
        
        Returns
        -------
        Ra : float
             Airy Radius as being 1.22 * WL * Fw/# in microns.
             Note that 3.8317059 / pi is closer to the real value 
             of the first zero of the sombrero function than 1.22.
        """
        return (3.8317059/np.pi) * self.wavelength * self.workingFNumber(offFieldX, offFieldY) * 1e6

    def wavefront(self, npx=64, background=False):
        """The function computes the wavefront error in image space with respect to reference sphere.

        Parameters
        ----------
        npx : int
             Size of the grid which corresponds to the number of pixels. Default is ``64`` meaning 64x64.
        background : bool
             If False then the output array is a numpy.ma.MaskedArray (no data outside of the pupil).
             If True then the output array is a numpy.array (0 where there is no data).
        
        Returns
        -------
        WF : array_like, shape (npx,npx)
             Wavefront error with respect to reference sphere.
             Note that the projection is set as ``zemax`` and reference as ``chief``.
             Plus the function returns minus the wavefront because of the orientation
             definition.
        """
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        #Rref = batoid.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1] - opt.items[-1].coordSys.origin[-1]
        XPpos = utils.exitPupilPos(optic=opt, wavelength=wl)
        Rref = XPpos[-1] - opt.items[-1].coordSys.origin[-1]
        WF = utils.wavefront(optic=opt,
                             theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                             wavelength=wl,
                             nx=npx,
                             sphereRadius=Rref,
                             projection='zemax',
                             reference='chief')
        
        if background:
            return (WF.array.data) * ~WF.array.mask
        else:
            return WF.array

    def wavefrontZnk(self, npx=64, jmax=12, eps=0, show=False):
        """The function computes Zernikes coefficients of the wavefront using Noll's definition.

        Parameters
        ----------
        npx : int
             Size of the grid which corresponds to the number of pixels. Default is ``64`` meaning 64x64.
        jmax : int
             Maximum Zernike coeficient. Default is ``12``.
        show : bool
             Displaying coefficients like in Zemax. Default is ``False``.
             Note that, for the moment, diplaying works for jmax <= 14.
        
        Returns
        -------
        Znk : array_like, len (jmax+1)
             Zernike coefficient in waves.
             Note that the first element is 0 so that Znk[i] corresponds to the i-th 
             Zernike coefficient as defined by Noll.
        """
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        #Rref = batoid.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1] - opt.items[-1].coordSys.origin[-1]
        XPpos = utils.exitPupilPos(optic=opt, wavelength=wl)
        Rref = XPpos[-1] - opt.items[-1].coordSys.origin[-1]
        #Znk = batoid.analysis.zernikeGQ(optic=opt,
         #                            theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
          #                           wavelength=wl,
           #                          rings=npx,
            #                         sphereRadius=Rref, 
             #                        projection='zemax',
              #                       reference='chief', 
               #                      jmax=jmax, eps=0.2)
        #Znk = batoid.analysis.zernike(optic=opt,
         #                          theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
          #                         wavelength=wl,
           #                        projection='zemax',
            #                       nx=npx,
             #                      sphereRadius=Rref,
              #                     reference='chief',
               #                    jmax=jmax, eps=0.2)
        Znk = utils.zernike(optic=opt,
                            theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                            wavelength=wl,
                            projection='zemax',
                            nx=npx,
                            sphereRadius=Rref,
                            reference='chief',
                            jmax=jmax, eps=eps)
        
        ## The orientation is diffrent to Zemax and so are how angles are defined.
        ## All spherical coefficients need to be multiplied by -1 + the angle is negative so sines as to be remultiplied by -1
        #Znk *= -1
        #val = np.linspace(0, len(Znk)-1, len(Znk), dtype='int') # -1 where SIN => negative angle
        #mask = np.logical_and.reduce((val%2==1, val>1, val!=11, val!=22, val!=37))
        #Znk[mask] *= -1

        if show:
            print(f'RMS (to chief)    :    {Znk[np.abs(Znk) > 1e-7].std():.9f} (waves)\n')
            for i in range(1, len(Znk)):
                if i<=9:
                    if Znk[i] >= 0:
                        print(f'{"Z":<4}{i:<7}{Znk[i]:.9f}{"":<2}{":":<7}{self.describe_zernike_polar(i)}')
                    else:
                        print(f'{"Z":<4}{i:<6}{Znk[i]:.9f}{"":<2}{":":<7}{self.describe_zernike_polar(i)}')
                else:
                    if Znk[i] >= 0:
                        print(f'{"Z":<3}{i:<8}{Znk[i]:.9f}{"":<2}{":":<7}{self.describe_zernike_polar(i)}') 
                    else:
                        print(f'{"Z":<3}{i:<7}{Znk[i]:.9f}{"":<2}{":":<7}{self.describe_zernike_polar(i)}')
            print('\n')
        return Znk

    def describe_zernike_polar(self, j):
        P, A = symbols('P A')

        out = galsim.zernike.describe_zernike(j)

        # Replace '^' with '**' for Python compatibility
        out = re.sub(r'\^', '**', out)
        # Add explicit multiplication between numbers and variables
        out = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', out)
        # Add explicit multiplication between variable pairs (e.g., 'xy' -> 'x*y') + avoid fct names
        out = re.sub(r'(?<![a-zA-Z])([a-zA-Z])([a-zA-Z])(?!\w)', r'\1*\2', out)
        # Replace 'x' and 'y' in the string with symbolic expressions
        out = re.sub(r'\bx\b', '(P*cos(A))', out)  # Match whole word 'x'
        out = re.sub(r'\by\b', '(P*sin(A))', out)  # Match whole word 'y'
        return str(trigsimp(out)).replace('**', '^')



 


    def EFFL(self):
        """The function computes the Effective Focal Length (EFFL) of the system.

        Parameters
        ----------
        
        Returns
        -------
        EFFL : float
            EFFL in meters.
        """
        return batoid.analysis.focalLength(optic=self.optic,
                                        theta_x=0, theta_y=0,
                                        wavelength=self.wavelength,
                                        projection='zemax')
    
    def posChiefRayAvg(self, surface=None):
        smallAngle = np.deg2rad(1./3600)

        thx = np.array([0, 0, smallAngle, -smallAngle]) + np.deg2rad(self.fieldX)
        thy = np.array([smallAngle, -smallAngle, 0, 0]) + np.deg2rad(self.fieldY)

        chiefRays = batoid.RayVector.fromFieldAngles(-thx, -thy, optic=self.optic, wavelength=self.wavelength) # minus angles bc not using custom zemaxToDirCos
        chiefRays = self.optic.traceFull(chiefRays, reverse=False)

        if surface is not None:
            try:
                globalChiefRays = chiefRays.get(surface).get('out').toCoordSys(batoid.globalCoordSys)
                print(globalChiefRays.r)
                r_chiefRay = np.mean(globalChiefRays.r, axis=0)
            except:
                raise Exception(f"Surface unavailable. Here's the acceptable surfaces : {(*chiefRays,)}")
        elif surface is None:
            listOfOpts = list(chiefRays.keys())
            lastOptName = listOfOpts[-1]
            globalChiefRays = chiefRays.get(lastOptName).get('out').toCoordSys(batoid.globalCoordSys)
            r_chiefRay = np.mean(globalChiefRays.r, axis=0)
        return r_chiefRay
    
    
    def posChiefRay(self, surface=None):
        """The function computes the position of the chief ray at a specific surface.

        Parameters
        ----------
        
        Returns
        -------
        x, y : float, float
            Position of the chief ray at a specific surface.
        """
        chiefRay = batoid.RayVector.fromStop(x=0, y=0,
                                          optic=self.optic,
                                          wavelength=self.wavelength,
                                          dirCos=utils.zemaxToDirCos(np.deg2rad(self.fieldX), np.deg2rad(self.fieldY)),
                                          flux=1.)
        chiefRay = self.optic.traceFull(chiefRay, reverse=False)

        if surface is not None:
            try:
                globalChiefRay = chiefRay.get(surface).get('out').toCoordSys(batoid.globalCoordSys)
                r_chiefRay = globalChiefRay.r[0]
            except:
                raise Exception(f"Surface unavailable. Here's the acceptable surfaces : {(*chiefRay,)}")
        elif surface is None:
            listOfOpts = list(chiefRay.keys())
            lastOptName = listOfOpts[-1]
            globalChiefRay = chiefRay.get(lastOptName).get('out').toCoordSys(batoid.globalCoordSys)
            r_chiefRay = globalChiefRay.r[0]
        return r_chiefRay

    def posChiefRayAtImgPlane(self):
        """The function computes the position of the chief ray in the image plane.

        Parameters
        ----------
        
        Returns
        -------
        x, y : float, float
            Position of the chief ray in the image plane.
        """
        return self.posChiefRay(surface=None)

    def strehlRatio(self):
        """The function computes the Strehl ratio of the system.
           The tilt free wavefront is along X and Y.

        Parameters
        ----------
        
        Returns
        -------
        strehlRation : float
            Strehl ration as being exp(-(2*WFrms)^2).
        """
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY

        dt = details(opt, fieldX=fieldX, fieldY=fieldY, wavelength=wl)
        WF = dt.wavefront(npx=1024)
        
        xp, yp = utils.getXY(opt, fieldX, fieldY, wl, 1024)
        Znk = dt.wavefrontZnk(npx=128, jmax=37, eps=0, show=False)

        # Znk = dt.wavefrontZnk(npx=128, jmax=37, show=False)
        corrWF, _ = correction.removeTiltWavefront(WF, Znk, xp, yp, removeTilt=False, removePiston=False)#setting removetilt to false
        RMS = corrWF.std()
        return np.exp(-(2*np.pi*RMS)**2)

    def huygensPSF(self, npx=32):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        
        StrehlRatio = self.strehlRatio()
        
        PSF = batoid.analysis.huygensPSF(optic=opt,
                                      wavelength=wl,
                                      theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                                      nx=npx,
                                      nxOut=npx,
                                      projection='zemax',
                                      reference='chief')
        PSF = PSF.array * StrehlRatio / np.max(PSF.array)
        return PSF, StrehlRatio

    def fftPSF(self, npx=16, pad=2):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY

        #StrehlRatio = self.strehlRatio()

        Rref = utils.exitPupilPos(optic=opt, wavelength=wl)[-1] - opt.items[-1].coordSys.origin[-1]
        
        PSF = batoid.analysis.fftPSF(optic=opt,
                                     wavelength=wl,
                                     theta_x=np.deg2rad(fieldX), theta_y=np.deg2rad(fieldY),
                                     nx=npx,
                                     pad_factor=pad,
                                     sphereRadius=Rref,
                                     projection='zemax',
                                     reference='chief')
        #PSF = PSF.array * StrehlRatio / np.max(PSF.array)
        #return PSF, StrehlRatio
        PSF = PSF.array / np.max(PSF.array)
        return PSF

    def imgPlaneLoc(self):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        D = opt.pupilSize
        
        dirCos = utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY))
        x, y = [D/2, 0, -D/2, 0], [0, D/2, 0, -D/2]
        rays = batoid.RayVector.fromStop(x=x, y=y, 
                                      optic=opt,
                                      wavelength=wl,
                                      dirCos=dirCos,
                                      flux=1.)

        opt.trace(rays, reverse=False)
        rays.toCoordSys(batoid.globalCoordSys)

        OA = batoid.RayVector.fromStop(x=0, y=0,
                                    optic=opt,
                                    wavelength=wl,
                                    dirCos=dirCos,
                                    flux=1.)
        
        opt.trace(OA, reverse=False)
        OA.toCoordSys(batoid.globalCoordSys)
        
        ps = []
        for ray in rays:
            ps.append(utils._closestApproach(ray.r[0], ray.v[0],
                                             OA.r[0], OA.v[0]
                                            )
                     )
        return np.mean(ps, axis=0)
    
    def spotDiagram(self, pattern='Hexapolar', density=1, source=None):
        return self.footPrint(surface=None, pattern=pattern, density=density, globalCoord=False, source=source)
    
    def footPrint(self, surface=None, pattern='Hexapolar', density=1, returnInitial=False, globalCoord=True, source=None):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY

        dirCos = utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY))
        
        nbOfRays = 1e3 * density

        if pattern.casefold() == 'dithered':
            #rays0 = ray(opt, fieldX=fieldX, fieldY=fieldY, wavelength=wl,
             #           nbOfRays=nbOfRays).fromPupil(type='grid')
            rays0 = batoid.RayVector.asGrid(optic=opt, wavelength=wl,
                                            dirCos=dirCos,
                                            nx=int(np.sqrt(nbOfRays)), ny=int(np.sqrt(nbOfRays)),
                                            nrandom=int(nbOfRays), rng=None)
        elif pattern.casefold() == 'hexapolar':
            rays0 = batoid.RayVector.asPolar(optic=opt, wavelength=wl, outer=opt.pupilSize/2,
                                             dirCos=dirCos,
                                             nrad=int(np.sqrt(2*nbOfRays)), naz=int(np.sqrt(2*nbOfRays)))
        elif pattern.casefold() == 'uniform':
            rays0 = batoid.RayVector.asGrid(optic=opt, wavelength=wl,
                                            dirCos=dirCos,
                                            nx=int(np.sqrt(nbOfRays)), ny=int(np.sqrt(nbOfRays)))
        else:
            raise Exception("Only 'Hexapolar', 'Dithered' and 'Uniform' patterns have been implemented yet.")
        
        if source is not None:
            x, y = rays0.x, rays0.y
            del rays0
            dirCos = utils.dirCosFromPointSource(source, utils.XYZ(x, y, opt.stopSurface))
            rays0 = batoid.RayVector.fromStop(x=x, y=y,
                                              optic=opt,
                                              wavelength=wl,
                                              dirCos=dirCos,
                                              flux=1.)
        rays = rays0.copy()

        output = opt.traceFull(rays, reverse=False)

        if surface is None: # None means last surface or image on detector
            rays = output[list(output)[-1]]['out']
            rays = ray.deleteVignettedRays(rays)

            if globalCoord:
                rays.toCoordSys(batoid.globalCoordSys)
            return (rays0, rays) if returnInitial else rays
        else:
            try:
                rays = output[surface]['out']
                rays = ray.deleteVignettedRays(rays)

                if globalCoord:
                    rays.toCoordSys(batoid.globalCoordSys)
                return (rays0, rays) if returnInitial else rays
            except:
                raise Exception(f"Surface unavailable. Here's the acceptable surfaces : {(*output,)}")
