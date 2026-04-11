import batoid
import numpy as np

from .rotation import rotation

class defineCoordinate:
    """Abstract base class defining coordinates for individual optical elements.
    """
    def __init__(self, posX=0, posY=0, posZ=0, anglX=0, anglY=0, anglZ=0):
        """
        Parameters
        ----------
        posX : float
            X position of the origin (in meters). Default is 0.
        posY : float
            Y position of the origin (in meters). Default is 0.
        posZ : float
            Z position of the origin (in meters). Default is 0.
        anglX : float
            Angle around X-axis (in degrees). Default is 0.
        anglY : float
            Angle around Y-axis (in degrees). Default is 0.
        anglZ : float
            Angle around Z-axis (in degrees). Default is 0.
        """
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.anglX = anglX
        self.anglY = anglY
        self.anglZ = anglZ
    
    def local(self, order='xyz'):
        """The function defining local coordinates for an individual optical element with a specific rotation order.

        Parameters
        ----------
        order : string, len (3)
            Order of the rotations to be applied.
            ``xyz`` (default), assumes we multiply rotation matrices in the following order: Rx * Ry * Rz

        Returns
        -------
        coordSys : batoid.CoordSys
             Coordinate system centered in (x,y,z) and rotated by (anglX,anglY,anglZ).
        """
        x, y, z = self.posX, self.posY, self.posZ
        R = rotation(self.anglX, self.anglY, self.anglZ)
        Rx, Ry, Rz = R.Rx(), R.Ry(), R.Rz()
        if order.casefold() == 'xyz':
            return batoid.CoordSys(origin=np.array([x, y, z]), rot=np.matmul(Rx, Ry, Rz))
        elif order.casefold() == 'yzx':
            return batoid.CoordSys(origin=np.array([x, y, z]), rot=np.matmul(Ry, Rz, Rx))
        elif order.casefold() == 'zyx':
            return batoid.CoordSys(origin=np.array([x, y, z]), rot=np.matmul(Rz, Ry, Rx))
