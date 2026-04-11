import numpy as np

class rotation:
    """Abstract base class defining 3D rotation matrices.
    """
    def __init__(self, anglX = 0., anglY = 0., anglZ = 0.):
        """
        Parameters
        ----------
        anglX : float
            Angle around X-axis (in degrees). Default is 0.
        anglY : float
            Angle around Y-axis (in degrees). Default is 0.
        anglZ : float
            Angle around Z-axis (in degrees). Default is 0.
        """
        self.anglX = anglX
        self.anglY = anglY
        self.anglZ = anglZ
    
    def Rx(self):
        """The function defining the 3D rotation matrix around X-axis; Rx(anglX).

        Parameters
        ----------
        anglX : float
            Angle of rotation in degree.

        Returns
        -------
        Rx : array_like, shape (3,3)
             Rotation matrix around X-axis.
        """
        theta = np.deg2rad(self.anglX)
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    
    def Ry(self):
        """The function defining the 3D rotation matrix around Y-axis; Ry(anglY).

        Parameters
        ----------
        anglY : float
            Angle of rotation in degree.

        Returns
        -------
        Ry : array_like, shape (3,3)
             Rotation matrix around Y-axis.
        """
        theta = np.deg2rad(self.anglY)
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    
    def Rz(self):
        """The function defining the 3D rotation matrix around Z-axis; Rz(anglZ).

        Parameters
        ----------
        anglZ : float
            Angle of rotation in degree.

        Returns
        -------
        Rz : array_like, shape (3,3)
             Rotation matrix around Z-axis.
        """
        theta: float = np.deg2rad(self.anglZ)
        return np.array([[np.cos(theta), -np.sin(theta),0 ],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    
    def rot(self, order='xyz'):
        if order.casefold() == 'xyz':
            return np.matmul(self.Rx(), self.Ry(), self.Rz())
        elif order.casefold() == 'yzx':
            return np.matmul(self.Ry(), self.Rz(), self.Rx())
        elif order.casefold() == 'zyx':
            return np.matmul(self.Rz(), self.Ry(), self.Rx())
        else:
            raise Exception(f"'{order}' does not correspond to any available possibilities.")
