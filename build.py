import batoid

from .definecoordinate import defineCoordinate

class build:
    """Abstract base class defining the building of an optical system.
       Note that, for now, it is only possible to define the stop as an entrance pupil.
    """
    def stopSurface(coordSys=defineCoordinate().local()):
        """The function defining the stop surface as being the entrance pupil.

        Parameters
        ----------
        coordSys : batoid.CoordSys
            Location and orientation of the stop. Defalt is ``(0,0,0);(1,0,0),(0,1,0),(0,0,1)``.
        
        Returns
        -------
        sys : batoid.Optic
             Interface of this stop.
        """
        return batoid.Interface(batoid.Plane(), coordSys=coordSys)
    
    def compoundOptic(*args, pupilSize, backDist=40, EPcoord):
        """The function defining a compound of optical surfaces as an optical system.

        Parameters
        ----------
        pupilSize : float
            Pupil's diameter (in meters).
        backDist : float
            Thickness of the first surface (in meters). Default is ``40``.
            Describes how far rays are coming to the first optical surface.
        
        Returns
        -------
        sys : batoid.Optic
             Compound of optical surfaces forming an optical system.
        """
        optics = [item for item in args]
        sys = batoid.CompoundOptic(optics,
                                backDist=backDist, 
                                stopSurface=build.stopSurface(coordSys=EPcoord),
                                pupilSize = pupilSize)
        return sys
