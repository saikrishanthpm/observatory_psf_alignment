import batoid
import numpy as np

from .definecoordinate import defineCoordinate
from .optic import optic
from .build import build

class utils:
    """Abstract base class redefining some specific native Batoid's function to match ZEMAX analysis.
    """   
    def __init__(self): 
        pass
        
    def zemaxToDirCos(u, v):
        """Convert Zemax field angles u,v to direction cosines.
    
        Parameters
        ----------
        u, v : float
            Zemax field angles in radians.
    
        Returns
        -------
        alpha, beta, gamma : float
            Direction cosines (unit vector projected onto x, y, z in order)
    
        Notes
        -----
        The tangent plane reference is at (u,v) = (0,0), which corresponds to
        (alpha, beta, gamma) = (0, 0, 1) (a ray coming directly from above).  The
        orientation is such that vx (vy) is positive when u (v) is negative.
    
        The Zemax field angle convention is not rotationally invariant.  The
        z-direction cosine for (u, v) = (0, 1) does not equal the z-direction
        cosine for (u, v) = (0.6, 0.8).
        """
        tanu = np.tan(u)
        tanv = np.tan(v)
        norm = np.sqrt(1 + tanu*tanu + tanv*tanv)
        return tanu/norm, tanv/norm, 1/norm

    def dirCosToZemax(alpha, beta, gamma):
        """Convert direction cosines to Postel azimuthal equidistant tangent plane
        projection.

        Parameters
        ----------
        alpha, beta, gamma : float
            Direction cosines (unit vector projected onto x, y, z in order)

        Returns
        -------
        u, v : float
            Postel tangent plane coordinates in radians.

        Notes
        -----
        The tangent plane reference is at (u,v) = (0,0), which corresponds to
        (alpha, beta, gamma) = (0, 0, +1) (a ray coming directly from above).  The
        orientation is such that vx (vy) is positive when u (v) is positive.

        The Zemax field angle convention is not rotationally invariant.  The
        z-direction cosine for (u, v) = (0, 1) does not equal the z-direction
        cosine for (u, v) = (0.6, 0.8).
        """
        return np.arctan(-alpha/gamma), np.arctan(-beta/gamma)

    def _closestApproach(P, u, Q, v):
        """Compute position along line P + u t of closest approach to line Q + v t

        Parameters
        ----------
        P, Q : ndarray
            Points on lines
        u, v : ndarray
            Direction cosines of lines

        Returns
        -------
        Pc : ndarray
            Closest approach point in meters.
        """
        # Follows http://geomalgorithms.com/a07-_distance.html
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        w0 = P - Q
        d = np.dot(u, w0)
        e = np.dot(v, w0)
        den = a*c - b*b
        if den == 0:
            raise ValueError("Lines are parallel")
        sc = (b*e - c*d)/den
        return P + sc*u

    def exitPupilPos(optic, wavelength, smallAngle=np.deg2rad(1./3600), reverse=False, **kwargs):
        """Compute position of the exit pupil.

        Traces a collection of small angle chief rays into object space, and then
        finds the mean of their closest approaches to the optic axis.  Possibly only
        accurate for axially symmetric optical systems.

        Parameters
        ----------
        optic : batoid.Optic
            Optical system
        wavelength : float
            Wavelength in meters
        smallAngle : float, optional
            Angle in radians from which to search for the exit pupil position.

        Returns
        -------
        Location of exit pupil in global coordinates (in meters).
        """
        thx = np.array([0, 0, smallAngle, -smallAngle])
        thy = np.array([smallAngle, -smallAngle, 0, 0])
        rays = batoid.RayVector.fromFieldAngles(thx, thy, optic=optic, wavelength=wavelength, **kwargs)
        optic.trace(rays, reverse=reverse)
        rays.toCoordSys(batoid.globalCoordSys)
        # Find the optical axis
        OA = batoid.RayVector.fromStop(x=0, y=0,
                                    optic=optic,
                                    wavelength=wavelength,
                                    dirCos=utils.zemaxToDirCos(np.deg2rad(0.0), np.deg2rad(0.0)),
                                    flux=1.)
        
        optic.trace(OA, reverse=reverse)
        OA.toCoordSys(batoid.globalCoordSys)
        # Assume last intersection places rays into "object" space.
        # Now find closest approach to optic axis.
        ps = []
        for ray in rays:
            ps.append(utils._closestApproach(ray.r[0], ray.v[0],
                                             OA.r[0], OA.v[0]
                                             #np.array([0., 0., 0.]),
                                             #np.array([0., 0., 1.0])
                                            )
                     )
        return np.mean(ps, axis=0)

    def exitPupilDir(optic, wavelength, smallAngle=np.deg2rad(1./3600), v=0.7, **kwargs):
        thx = np.array([0, 0, smallAngle, -smallAngle])
        thy = np.array([smallAngle, -smallAngle, 0, 0])
        R = optic.pupilSize / 2
        rays = batoid.RayVector.fromFieldAngles(thx, thy, x=v*R/np.sqrt(2), y=v*R/np.sqrt(2),
                                             optic=optic, wavelength=wavelength, **kwargs)
        optic.trace(rays)
        rays.toCoordSys(batoid.globalCoordSys)

        center = batoid.RayVector.fromFieldAngles(0, 0, x=v*R/np.sqrt(2), y=v*R/np.sqrt(2), 
                                               optic=optic, wavelength=wavelength, **kwargs)
        optic.trace(center)
        center.toCoordSys(batoid.globalCoordSys)

        # Assume last intersection places rays into "object" space.
        # Now find closest approach to optic axis.
        ps = []
        for ray in rays:
            ps.append(utils._closestApproach(ray.r[0], ray.v[0],
                                             center.r[0], center.v[0]
                                            )
                     )
        posInXP = np.mean(ps, axis=0)
        refPosinXP = utils.exitPupilPos(optic, wavelength, **kwargs)

        r = np.sqrt((posInXP[0]-refPosinXP[0])**2 + (posInXP[1]-refPosinXP[1])**2)
        theta = np.arctan((posInXP[-1]-refPosinXP[-1]) / r)
        print(np.rad2deg(theta))
        return posInXP
        

    def XPdiam(opt, theta_x=0, theta_y=0, wavelength=633e-9, reverse=False):
        nx = 256
        dirCos = utils.zemaxToDirCos(np.deg2rad(theta_x), np.deg2rad(theta_y))
        #rays = batoid.RayVector.asGrid(optic=opt, wavelength=wavelength, nx=nx, dirCos=dirCos)
        DD = opt.pupilSize
        th = np.linspace(0, 2*np.pi, nx)
        x, y = np.cos(th)*DD/2, np.sin(th)*DD/2
        rays = batoid.RayVector.fromStop(x=x, y=y, optic=opt, wavelength=wavelength, dirCos=dirCos)
        
        XPloc = utils.exitPupilPos(opt, wavelength=wavelength, reverse=reverse)
        XPcoord = defineCoordinate(posX=XPloc[0], posY=XPloc[1], posZ=XPloc[-1]).local()
        originalD = opt.items[-1]
        XP = optic('XP', inDiam=originalD.inDiam, outDiam=originalD.outDiam, coordSys=XPcoord).detector()
        newSysAtXP = build.compoundOptic(*opt.items[0:-1], XP,
                                         pupilSize=opt.pupilSize, backDist=opt.backDist,
                                         EPcoord=opt.stopSurface.coordSys)
        newSysAtXP.trace(rays, reverse=reverse)
        rays.toCoordSys(batoid.globalCoordSys)
        
        #w = np.where(1-rays.vignetted)[0]

        ##x, y = rays.x[w]-np.mean(rays.x[w]), rays.y[w]-np.mean(rays.y[w])
        x, y = rays.x-np.mean(rays.x), rays.y-np.mean(rays.y)
        r = np.sqrt(x**2 + y**2)
        b = 2 * np.mean(r)
        
        return b

    def pupilsAnalysis(entranceSys, exitSys, fieldX=0, fieldY=0, wavelength=633e-9):

        dict = {}
        
        XPdiam = utils.XPdiam(entranceSys, theta_x=fieldX, theta_y=fieldY, wavelength=wavelength, reverse=True)
        XPpos = utils.exitPupilPos(optic=entranceSys, wavelength=wavelength, reverse=True)
        Rref = XPpos[-1] + entranceSys.backDist

        dict.update({"ENPP": {"abs pos":XPpos[-1], "rel pos":Rref, "diam":XPdiam}})

        XPdiam = utils.XPdiam(exitSys, theta_x=fieldX, theta_y=fieldY, wavelength=wavelength, reverse=False)
        XPpos = utils.exitPupilPos(optic=exitSys, wavelength=wavelength, reverse=False)
        Rref = XPpos[-1] - exitSys.items[-1].coordSys.origin[-1]

        dict.update({"EXPP": {"abs pos":XPpos[-1], "rel pos":Rref, "diam":XPdiam}})

        return dict

    def zernike(optic, theta_x, theta_y, wavelength,
                projection='postel', nx=32,
                sphereRadius=None, reference='chief', jmax=22, eps=0.0):
        import galsim

        #dirCos = batoid.utils.fieldToDirCos(theta_x, theta_y, projection=projection)
        dirCos = utils.zemaxToDirCos(theta_x, theta_y)
        rays = batoid.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                    nx=nx, dirCos=dirCos)
        
        # Propagate to entrance pupil to get positions
        epRays = rays.toCoordSys(optic.stopSurface.coordSys)
        optic.stopSurface.surface.intersect(epRays)
        orig_x = np.array(epRays.x).reshape(nx, nx)
        orig_y = np.array(epRays.y).reshape(nx, nx)

        wf = utils.wavefront(optic, theta_x, theta_y, wavelength, nx=nx,
                             projection=projection, sphereRadius=sphereRadius,
                             reference=reference)
        wfarr = wf.array
        w = ~wfarr.mask

        basis = galsim.zernike.zernikeBasis(jmax, orig_x[w], orig_y[w],
                                            R_outer=optic.pupilSize/2, R_inner=optic.pupilSize/2*eps
                                           )
        #plt.figure()
        #plt.scatter(orig_x[w]/(optic.pupilSize/2), orig_y[w]/(optic.pupilSize/2))
        
        coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=-1)
        # coefs[0] is meaningless, so always set to 0.0 for comparison consistency
        coefs[0] = 0.0
        return np.array(coefs)

    def getXY(optic, theta_x, theta_y, wavelength, nx):
        #dirCos = batoid.utils.fieldToDirCos(theta_x, theta_y, projection=projection)
        dirCos = utils.zemaxToDirCos(theta_x, theta_y)
        rays = batoid.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                    nx=nx, dirCos=dirCos)
        
        # Propagate to entrance pupil to get positions
        epRays = rays.toCoordSys(optic.stopSurface.coordSys)
        optic.stopSurface.surface.intersect(epRays)
        orig_x = np.array(epRays.x).reshape(nx, nx)
        orig_y = np.array(epRays.y).reshape(nx, nx)

        return orig_x/(optic.pupilSize/2), orig_y/(optic.pupilSize/2)

    def zernikeFromWavefront(WF, x, y, jmax=22):
        import galsim

        wfarr = WF.data
        w = ~WF.mask

        #N = np.shape(wfarr)[0]

        #val = np.linspace(-1, 1, N)
        #x, y = np.meshgrid(val, val)

        
        basis = galsim.zernike.zernikeBasis(jmax, x[w], y[w],
                                            R_outer=1, R_inner=0
                                           )
        #plt.scatter(x[w], y[w])
        #plt.axis('equal')
        #plt.show()
        
        coefs, _, _, _ = np.linalg.lstsq(basis.T, wfarr[w], rcond=-1)
        # coefs[0] is meaningless, so always set to 0.0 for comparison consistency
        coefs[0] = 0.0
        return np.array(coefs)

    def wavefrontFromZernike(Znk, x, y, mask, nx):
        import galsim
        
        W = galsim.zernike.Zernike(coef=Znk).evalCartesian(x, y)
        W = np.ma.masked_array(data=W, mask=mask)
        
        return W

    ########

    def wavefront(optic, theta_x, theta_y, wavelength,
                  projection='postel', nx=32, sphereRadius=None, reference='chief'):
        
        dirCos = utils.zemaxToDirCos(theta_x, theta_y)
        rays = batoid.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                    nx=nx, dirCos=dirCos)
        
        if sphereRadius is None:
            sphereRadius = optic.sphereRadius

        optic.trace(rays)
        
        cridx, point = utils.referencePoint(rays, nx=nx)
            
        # Place vertex of reference sphere one radius length away from the
        # intersection point.  So transform our rays into that coordinate system.
        targetCoordSys = rays.coordSys.shiftLocal(point+np.array([0, 0, sphereRadius])
                                                 )
        rays.toCoordSys(targetCoordSys)

        sphere = batoid.Sphere(-sphereRadius)
        sphere.intersect(rays)

        t0 = rays.t[cridx]
        OPD = (t0-rays.t)/wavelength
            
        arr = np.ma.masked_array(OPD,
                                 mask=rays.vignetted).reshape(nx, nx)
        if (nx%2) == 0:
            primitiveU = np.vstack([[optic.pupilSize/(nx-2), 0],
                                    [0, optic.pupilSize/(nx-2)]])
        else:
            primitiveU = np.vstack([[optic.pupilSize/(nx-1), 0],
                                    [0, optic.pupilSize/(nx-1)]])
        return batoid.Lattice(arr, primitiveU)
        
    ########
    ########
    def referencePoint(rays, dx=0, dy=0, dz=0, 
                       nx=32):
        
        cridx = (nx//2)*nx+nx//2 if (nx%2)==0 else (nx*nx-1)//2
        point = rays.r[cridx]
        
        return cridx, point + np.array([dx, dy, dz])
    
    # def dirCosFromPoint(p1, p2):
    #     p1 = np.asarray(p1)
    #     p2 = np.asarray(p2)
    #     dirVec = p2 - p1
    #     mag = np.linalg.norm(dirVec)
    #     if mag == 0:
    #         raise ValueError('The two points are identical; direction cosines cannot be determined.')
    #     return dirVec / mag
    
    def dirCosFromPointSource(p1, p2_list):
        """
        Compute the direction cosines from a single point p1 to multiple points in p2_list.

        Parameters:
            p1 (array-like): A 3-element array [x, y, z] representing a single point.
            p2_list (array-like or np.ndarray): Either:
                - A (3,) array representing a single point [x, y, z]
                - A (3, N) array where each row is [x_list, y_list, z_list]

        Returns:
            numpy.ndarray: A (3, N) array of unit direction vectors, or (3,) if p2_list is a single point.
        """
        p1 = np.asarray(p1)  # Convert to NumPy array
        p2_list = np.asarray(p2_list)  # Convert p2_list to a NumPy array

        # If p2_list is a single point (1D array), reshape to (3,1) for correct broadcasting
        if p2_list.shape == (3,):
            p2_list = p2_list[:, np.newaxis]  # Reshape to (3,1) so broadcasting works

        # Ensure p2_list is in (3, N) format
        if p2_list.shape[0] != 3:
            raise ValueError("p2_list must have shape (3,) for a single point or (3, N) for multiple points.")

        # Compute direction vectors (vectorized)
        dirVecs = p2_list - p1[:, np.newaxis]  # Shape: (3, N)

        # Compute magnitudes along each column
        mags = np.linalg.norm(dirVecs, axis=0, keepdims=True)  # Shape: (1, N)

        if np.any(mags == 0):
            raise ValueError("At least one point in p2_list is identical to p1; direction cosines cannot be determined.")

        # Normalize direction vectors correctly
        result = dirVecs / mags  # Shape remains (3, N)

        # If original input was a single point, return (3,) instead of (3,1)
        return result.flatten() if p2_list.shape[1] == 1 else result
        
    # def XYZ(x, y, stopSurface):
    #     x = np.atleast_1d(x).astype(float, copy=True)
    #     y = np.atleast_1d(y).astype(float, copy=True)
    #     z = stopSurface.surface.sag(x, y)
    #     transform = batoid.CoordTransform(stopSurface.coordSys, batoid.globalCoordSys)
    #     batoid.trace.applyForwardTransformArrays(transform, x, y, z)
    #     return np.array([x, y, z]).flatten()

    def XYZ(x, y, stopSurface):
        """
        Compute the (X, Y, Z) coordinates from input x, y values using the stopSurface.

        Parameters:
            x (float or array-like): X-coordinate(s), can be a scalar or list/array.
            y (float or array-like): Y-coordinate(s), can be a scalar or list/array.
            stopSurface (object): Surface object containing the sag function and coordinate system.

        Returns:
            numpy.ndarray: A (3, N) array where:
                - First row contains x values,
                - Second row contains y values,
                - Third row contains corresponding z values.
        """
        # Ensure inputs are arrays, even for single values
        x = np.atleast_1d(x).astype(float, copy=True)
        y = np.atleast_1d(y).astype(float, copy=True)

        # Validate input sizes match
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        # Compute the sag (Z-coordinates)
        z = stopSurface.surface.sag(x, y)  # Should return same shape as x, y

        # Apply coordinate transformation
        transform = batoid.CoordTransform(stopSurface.coordSys, batoid.globalCoordSys)
        batoid.trace.applyForwardTransformArrays(transform, x, y, z)

        # Stack results into (3, N) format, preserving the same number of x, y, z values
        result = np.vstack([x, y, z])

        return result  # No flattening to maintain (3, N) format
