import batoid
import numpy as np

from .utils import utils

class ray:
    """Abstract base class defining different ray possibilities and computations.
    """
    def __init__(self,
                 opticalSys, fieldX=0, fieldY=0, wavelength=633e-9, nbOfRays=1
                ):
        """
        Parameters
        ----------
        opticalSys : batoid.Optic
            Optical system as a single optic or a compound optic.
        fieldX : float
            Field over X (in degrees). Default is ``0``.
        fieldY : float
            Field over Y (in degrees). Default is ``0``.
        wavelength : float
            Wavelength for this ray (in meters). Default is ``633e-9``.
        nbOfRays : integer
            Number of rays to create. Default is ``1``.
        """
        self.opticalSys = opticalSys
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.wavelength = wavelength
        self.nbOfRays = nbOfRays

    def createPolarBundleOfRays(self):
        """The function defining x and y coordinates of each ray.
           This is a uniform distribution of rays over angles and radii of the pupil.

        Parameters
        ----------
        
        Returns
        -------
        x : array_like
             X coordinates.
        y : array_like
             Y coordinates.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        th = 2 * np.pi * np.random.rand(int(N))
        cth, sth = np.cos(th), np.sin(th)
        d = (size) * np.random.rand(int(N))
        x, y = (d/2) * cth, (d/2) * sth
        return x, y

    def createGridBundleOfRays(self):
        """The function defining x and y coordinates of each ray.
           This is a uniform distribution of rays over the tangential and sagittal directions of the pupil.

        Parameters
        ----------
        
        Returns
        -------
        x : array_like
             X coordinates.
        y : array_like
             Y coordinates.
        """
        N, size = int((4/np.pi) * self.nbOfRays), self.opticalSys.pupilSize
        xi = np.random.uniform(-size/2, size/2, (int(np.sqrt(N)), int(np.sqrt(N))))
        yi = np.random.uniform(-size/2, size/2, (int(np.sqrt(N)), int(np.sqrt(N))))

        indx = (xi**2 + yi**2 <= (size/2)**2)
        x, y = xi[indx], yi[indx]
        points = np.row_stack((x.flatten(), y.flatten()))
        del x, y
        x, y = points[0, :], points[1, :]
        
        return x, y

    def fromPupil(self, type='Polar'):
        """The function defining rays all over the pupil. This is uniformly random.

        Parameters
        ----------
        type : string
             'Polar' for a uniform polar distribution. This is default.
             'Grid' for a uniform grid distribution.
        
        Returns
        -------
        rays : batoid.RayVector
             Defining a uniform bundle of rays all over the pupil.
        """
        if type.casefold() == 'polar':
            x, y = self.createPolarBundleOfRays()
        elif type.casefold() == 'grid':
            x, y = self.createGridBundleOfRays()
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = batoid.RayVector.fromStop(x=x, y=y, 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY)),
                                      flux=1.)
        return rays

    def fromPupilSingle(self, r):
        """The function defining rays all over the pupil. This is a user defined bundle.

        Parameters
        ----------
        r : list
             User defined bundle of ray(s). It can be one or several ones.
             For instance, r=[0, 0] will produce a single centered ray while
             r=[np.zeros(N), np.linspace(-R, R, N)] will produce a bundle of ray
             between -R and R on the Y axis.
        
        Returns
        -------
        rays : batoid.RayVector
             Defining a bundle of rays all over the pupil.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = batoid.RayVector.fromStop(x=r[0], y=r[1], 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY)),
                                      flux=1.)
        return rays
        
    def fromPupilForFan(self, axis='y'):
        """The function defining rays all over the pupil fro plotting a Ray Fan.

        Parameters
        ----------
        axis : string (len=1)
             Can only take 'x' and 'y' depeding if we want a uniform distribution over X or Y.
        
        Returns
        -------
        rays : batoid.RayVector
             Defining a bundle of rays all over the pupil.
        """
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        if axis.casefold() == 'y':
            x = np.zeros(int(N))
            y = np.linspace(-size, size, int(N)) / 2
        elif axis.casefold() == 'x':
            y = np.zeros(int(N))
            x = np.linspace(-size, size, int(N)) / 2
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = batoid.RayVector.fromStop(x=x, y=y, 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=utils.zemaxToDirCos(np.deg2rad(fieldX), np.deg2rad(fieldY)),
                                      flux=1.)
        return rays

    def deleteVignettedRays(rv):
        """The function deleting vignetted rays (the ones being stopped by obscurations).
           Use rays being ".trace" not ".traceFull" or ".traceSplit".

        Parameters
        ----------
        rv : batoid.rayVector
             Bundle of rays to analyze and potentially delete.
        
        Returns
        -------
        rv : batoid.RayVector
             Keep unvignetted rays.
        """
        which = np.where(1-rv.vignetted)[0]
        #which = np.where(np.invert(rv.vignetted))[0].tolist()
        return rv[which]

    def intersection(self, rays):
        """The function computing the intersections between each surface.
           Use rays being ".traceFull" not ".trace" or ".traceSplit".

        Parameters
        ----------
        rays : batoid.RayVector
             Bundle of rays being trace at each surface meaning using ".traceFull".
        
        Returns
        -------
        intersec_dict : dict
             Dictionnary of array_like at each surface.
             For instance if surface 1 is called "S1", then we use ".get('S1')" to retrive 
             the array_like of all the rays with the surface ([x,y,z]).
        """
        optic = self.opticalSys
        listOfOpts = list(rays.keys())
        intersec_dict = dict()
        args = optic.items
        for i in range(len(listOfOpts)):
            optName = listOfOpts[i]
            
            r = rays.get(optName).get('out').r
            intersec_dict[optName] = batoid.CoordTransform(args[i].coordSys, 
                                                        batoid.CoordSys(origin=np.array([0, 0, 0]),
                                                                     rot=np.identity(3))).applyForwardArray(r[:, 0],
                                                                                                            r[:, 1],
                                                                                                            r[:, 2]).transpose()
        return intersec_dict

    def directionCos(self, rays):
        """The function computing the direction cosine vectors before and after each surface.
           Use rays being ".traceFull" not ".trace" or ".traceSplit".

        Parameters
        ----------
        rays : batoid.RayVector
             Bundle of rays being trace at each surface meaning using ".traceFull".
        
        Returns
        -------
        dir_dict : dict
             Dictionnary of array_like at each surface, before and after.
             For instance if surface 1 is called "S1", then we use ".get('S1').get('in')" to retrive 
             the array_like of all the directions before the surface. ".get('S1').get('out')" for the ones after.
        """
        optic, wl = self.opticalSys, self.wavelength
        listOfOpts = list(rays.keys())
        dir_dict = dict()
        args = optic.items
        for i in range(len(listOfOpts)):
            optName = listOfOpts[i]
            nIN, nOUT = args[i].inMedium.getN(wl), args[i].outMedium.getN(wl)
            
            vIN = rays.get(optName).get('in').v * nIN
            vOUT = rays.get(optName).get('out').v * nOUT
            dir_dict[optName] = {'in':vIN, 'out':vOUT}
        return dir_dict
