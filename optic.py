import batoid
import numpy as np

from .definecoordinate import defineCoordinate

class optic:
    """Abstract base class defining some optical elements (only detectors and mirrors for now).
    """
    def __init__(self, name, type='Plane', 
                 RoC=-1, conic=0, inDiam=0, outDiam=1, indx=1.0,
                 coordSys=defineCoordinate().local()
                ):
        """
        Parameters
        ----------
        name : string
            Name of the optical surface.
        type : string
            Defining the shape of the surface. Default is ``Plane``.
        RoC : float
            Radius of curvature of the surface (in meters). Default is ``-1``.
        conic : float
            Conic constant of the surface. Default is ``0``.
        inDiam : float
            Inner diameter of the surface (in meters). Default is ``0``.
        outDiam : float
            Outter diameter of the surface (in meters). Default is ``1``.
        indx : float
            Refractive index before and after the surface. Default is ``1.0``.
        coordSys : batoid.CoordSys
            Coordinate system associated with the surface. Default is ``(0,0,0);(1,0,0),(0,1,0),(0,0,1)``.
        """
        self.name = name
        self.type = type
        self.RoC = RoC
        self.conic = conic
        self.inDiam = inDiam
        self.outDiam = outDiam
        self.indx = indx
        self.coordSys = coordSys

    def obscMirror(self):
        """The function defining the oscuration of a mirror.

        Parameters
        ----------
        
        Returns
        -------
        coordSys : batoid.Obscuration
             Annulus light going through depending on inDiam and outDiam. 
             If inDiam=0, then the light goes through a circle of a diameter of outDiam.
             ``+10`` represent the upper bound of obscuration's circle. It is arbitrary but it avoids straylight.
        """
        inDiam, outDiam = self.inDiam, self.outDiam
        if inDiam == 0:
            return batoid.ObscAnnulus(outDiam/2, outDiam/2+10)
        else:
            return batoid.ObscUnion(batoid.ObscCircle(inDiam/2), batoid.ObscAnnulus(outDiam/2, outDiam/2+10))

    # def obscMirrorWithSpider(self, width, height):
    #     """The function defining the oscuration of a mirror with a 3-arms spider.

    #     Parameters
    #     ----------
    #     width : float
    #         Arms' width (in meters).
    #     height : float
    #         Arms' height (in meters).
        
    #     Returns
    #     -------
    #     coordSys : batoid.Obscuration
    #          Annulus light going through depending on inDiam and outDiam.
    #          3 arms obscur this annulus.
    #     """
    #     obsc = batoid.ObscUnion(self.obscMirror(), 
    #                          batoid.ObscRectangle(width=width, height=height, 
    #                                            x=height/2, y=0.0, theta=np.pi/2),
    #                          batoid.ObscRectangle(width=width, height=height, 
    #                                            x=-height/2*np.sin(np.pi/6), y=-4/2*np.cos(np.pi/6), theta=-np.pi/6),
    #                          batoid.ObscRectangle(width=width, height=height, 
    #                                            x=-height/2*np.sin(5*np.pi/6), y=-4/2*np.cos(5*np.pi/6), theta=-5*np.pi/6))
    #     return obsc

    def obscMirrorWithSpider(self, num_arms, arm_width, theta_offset=0):
        """
        Create a spider obscuration combined with the native mirror obscuration.

        Parameters:
        -----------
        num_arms : int
            Number of spider arms.
        arm_width : float
            Width of each spider arm (in the same units as the radius).
        theta_offset : float, optional
            Angular offset for rotating the spider arms, in radians. Default is 0.

        Returns:
        --------
        batoid.Obscuration
            A batoid.Obscuration object representing the combined obscuration.
        """
        # List to store spider arm obscurations
        arms = []

        radius = self.outDiam/2

        # Angle between adjacent arms
        delta_theta = 2 * np.pi / num_arms

        for i in range(num_arms):
            # Angle of the current arm
            theta = i * delta_theta + theta_offset

            # Calculate the arm's center position (rotated by theta)
            x_center = np.cos(theta) * radius / 2
            y_center = np.sin(theta) * radius / 2

            # Create a rectangular arm
            arm = batoid.ObscRectangle(
                width=radius,         # Length of the arm (diameter of the mirror)
                height=arm_width,     # Width of the arm
                x=x_center,
                y=y_center,
                theta=theta           # Rotation angle
            )

            # Add the arm to the list
            arms.append(arm)

        # Combine spider arms using ObsUnion
        spider_obsc = batoid.ObscUnion(*arms)

        # Combine spider obscuration with the native mirror obscuration
        combined_obsc = batoid.ObscUnion(self.obscMirror(), spider_obsc)

        return combined_obsc
            
    def flatMirror(self):
        """The function defining a flat mirror.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Flat mirror.
        """
        type = self.type
        if type.casefold() == 'plane':
            obj = batoid.Mirror(name=self.name,
                             surface=batoid.Plane(), 
                             obscuration=self.obscMirror(),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=batoid.ConstMedium(self.indx),
                             outMedium=batoid.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj    

    def mirror(self):
        """The function defining a simple mirror with an annulus bundle of rays.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Mirror following a ``Quadric`` desciption, meaning RoC + conic.
        """
        type = self.type 
        if type.casefold() == 'quadric':
            if self.conic == -1:
                obj = batoid.Mirror(name=self.name,
                                surface=batoid.Paraboloid(self.RoC), 
                                obscuration=self.obscMirror(),
                                inDiam=self.inDiam,
                                outDiam=self.outDiam,
                                inMedium=batoid.ConstMedium(self.indx),
                                outMedium=batoid.ConstMedium(self.indx),
                                coordSys=self.coordSys,
                                skip=False)
            else:
                obj = batoid.Mirror(name=self.name,
                                surface=batoid.Quadric(self.RoC, self.conic), 
                                obscuration=self.obscMirror(),
                                inDiam=self.inDiam,
                                outDiam=self.outDiam,
                                inMedium=batoid.ConstMedium(self.indx),
                                outMedium=batoid.ConstMedium(self.indx),
                                coordSys=self.coordSys,
                                skip=False)
            return obj 

    # def mirrorWithSpider(self, width=0, height=0):
    #     """The function defining a simple mirror with an spider.

    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
    #     obj : batoid.Optic
    #          Mirror following a ``Quadric`` desciption, meaning RoC + conic.
    #     """
    #     type = self.type 
    #     if type.casefold() == 'quadric':
    #         obj = batoid.Mirror(name=self.name,
    #                          surface=batoid.Quadric(self.RoC, self.conic), 
    #                          obscuration=self.obscMirrorWithSpider(width, height),
    #                          inDiam=self.inDiam,
    #                          outDiam=self.outDiam,
    #                          inMedium=batoid.ConstMedium(self.indx),
    #                          outMedium=batoid.ConstMedium(self.indx),
    #                          coordSys=self.coordSys,
    #                          skip=False)
    #         return obj 

    def mirrorWithSpider(self, num_arms, arm_width, theta_offset=0):
        """The function defining a simple mirror with an spider.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Mirror following a ``Quadric`` desciption, meaning RoC + conic.
        """
        type = self.type 
        if type.casefold() == 'quadric':
            obj = batoid.Mirror(name=self.name,
                             surface=batoid.Quadric(self.RoC, self.conic), 
                             obscuration=self.obscMirrorWithSpider(num_arms, arm_width, theta_offset=theta_offset),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=batoid.ConstMedium(self.indx),
                             outMedium=batoid.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj

    def detector(self):
        """The function defining a detector. The detector should be the last surface in optical systems.

        Parameters
        ----------
        
        Returns
        -------
        obj : batoid.Optic
             Detector. It should be the last surface in optical systems.
        """
        type = self.type
        if type.casefold() == 'plane':
            obj = batoid.Detector(batoid.Plane(), 
                               coordSys=self.coordSys, 
                               name=self.name,
                               inDiam=self.inDiam,
                               outDiam=self.outDiam)
            return obj
