import batoid
import numpy as np
import pandas as pd
from itertools import chain
import plotly.express as px

from .utils import utils

class layout:

    def sag2DY(optic, inDiam, outDiam):
        N = 0.5e2
        if inDiam != 0:
            y1 = np.linspace(-outDiam, -inDiam, int(N/2))/2
            x1 = np.zeros(len(y1))
            z1 = optic.surface.sag(x1, y1)
            x1, y1, z1 = batoid.CoordTransform(optic.coordSys, 
                                            batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x1, y1, z1)
            y2 = np.linspace(inDiam, outDiam, int(N/2))/2
            x2 = np.zeros(len(y2))
            z2 = optic.surface.sag(x2, y2)
            x2, y2, z2 = batoid.CoordTransform(optic.coordSys, 
                                            batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x2, y2, z2)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            y = np.linspace(-outDiam, outDiam, int(N))/2
            x = np.zeros(len(y))
            z = optic.surface.sag(x, y)
            x, y, z = batoid.CoordTransform(optic.coordSys,
                                         batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x, y, z)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df
        
    def sag2DX(optic, inDiam, outDiam):
        N = 0.5e2
        if inDiam != 0:
            x1 = np.linspace(-outDiam, -inDiam, int(N/2))/2
            y1 = np.zeros(len(x1))
            z1 = optic.surface.sag(x1, y1)
            x1, y1, z1 = batoid.CoordTransform(optic.coordSys, 
                                            batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x1, y1, z1)
            x2 = np.linspace(inDiam, outDiam, int(N/2))/2
            y2 = np.zeros(len(x2))
            z2 = optic.surface.sag(x2, y2)
            x2, y2, z2 = batoid.CoordTransform(optic.coordSys, 
                                            batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x2, y2, z2)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            x = np.linspace(-outDiam, outDiam, int(N))/2
            y = np.zeros(len(x))
            z = optic.surface.sag(x, y)
            x, y, z = batoid.CoordTransform(optic.coordSys,
                                         batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x, y, z)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df 

    def sag2Dcircle(optic, inDiam, outDiam):
        N = 50
        if inDiam != 0:
            th = np.linspace(0, 2*np.pi, int(N))
            x1 = outDiam/2 * np.cos(th)
            y1 = outDiam/2 * np.sin(th)
            z1 = optic.surface.sag(x1, y1)
            x1, y1, z1 = batoid.CoordTransform(optic.coordSys, 
                                            batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x1, y1, z1)
            x2 = inDiam/2 * np.cos(th)
            y2 = inDiam/2 * np.sin(th)
            z2 = optic.surface.sag(x2, y2)
            x2, y2, z2 = batoid.CoordTransform(optic.coordSys, 
                                            batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x2, y2, z2)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            th = np.linspace(0, 2*np.pi, int(N))
            x = outDiam/2 * np.cos(th)
            y = outDiam/2 * np.sin(th)
            z = optic.surface.sag(x, y)
            x, y, z = batoid.CoordTransform(optic.coordSys,
                                         batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(x, y, z)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df

    def getMirrorLines(opt, inner, outter):
        if inner != 0:
            df1Y, df2Y = layout.sag2DY(opt, inner, outter)
            df1X, df2X = layout.sag2DX(opt, inner, outter)
            df1c, df2c = layout.sag2Dcircle(opt, inner, outter)
            return df1Y, df2Y, df1X, df2X, df1c, df2c
        else: 
            dfY = layout.sag2DY(opt, 0, outter)
            dfX = layout.sag2DX(opt, 0, outter)
            dfc = layout.sag2Dcircle(opt, 0, outter)
            return dfY, dfX, dfc

    def getAllLines(optic):
        N = len(optic.items)
        allLines = [0]* N
        for i in range(N):
            opt = optic.items[i]
            allLines[i] = layout.getMirrorLines(opt, opt.inDiam, opt.outDiam)
        return chain(*allLines)

    def rayBundle(optic, WL, inDiam, outDiam, anglX, anglY, nbOfRays, source=None):
        if nbOfRays == 1:
            y = [0]
        else:
            if inDiam != 0:
                y = np.concatenate((np.linspace(-outDiam, -inDiam, int(nbOfRays/2)), np.linspace(inDiam, outDiam, int(nbOfRays/2)))) / 2
            else:
                y = np.linspace(-outDiam, outDiam, int(nbOfRays)) / 2
        rays = [0]*len(y)
        for i in range(len(y)):
            rays[i] = layout.getRay(optic, WL=WL, x=0, y=y[i], anglX=anglX, anglY=anglY, source=source)
        return rays

    def getRay(optic, WL, x=0, y=0, anglX=0, anglY=0, source=None):
        args = optic.items
        if source is not None:
            dirCos = utils.dirCosFromPointSource(source, utils.XYZ(x, y, optic.stopSurface))
        else:
            dirCos=utils.zemaxToDirCos(np.deg2rad(anglX), np.deg2rad(anglY))
        ray = batoid.RayVector.fromStop(x=x, y=y,
                                        optic=optic,
                                        wavelength=WL,
                                        dirCos=dirCos,
                                        flux=1.)
        ray = optic.traceFull(ray, reverse=False)
        listOfOpts = list(ray.keys())
        wholeRayTracing = [0] * len(args)
        for i in range(len(listOfOpts)):
            optName = listOfOpts[i]
            rIN, rOUT = ray.get(optName).get('in').r[0], ray.get(optName).get('out').r[0]
            #
            #vIN, vOUT = ray.get(optName).get('in').v[0], ray.get(optName).get('out').v[0]
            #if (i+1) % 2 == 0:
            #    print(i+1, 180-np.arccos(np.dot(vIN, np.array([0., 0., 1.])))*180/np.pi, np.arccos(np.dot(vOUT, np.array([0., 0., 1.])))*180/np.pi)
            #else:
            #     print(i+1, np.arccos(np.dot(vIN, np.array([0., 0., 1.])))*180/np.pi, 180-np.arccos(np.dot(vOUT, np.array([0., 0., 1.])))*180/np.pi)
            #
            coordSys = args[i].coordSys
            if i == 0:
                x, y, z = batoid.CoordTransform(coordSys,
                                             batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(rOUT[0], rOUT[1], rOUT[2])
                wholeRayTracing[i] = np.array([[rIN[0], x], [rIN[1], y], [rIN[2], z]])
            else:
                coordSysp = args[i-1].coordSys
                r = batoid.CoordTransform(coordSys,
                                       batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(rOUT[0],
                                                                                                                       rOUT[1],
                                                                                                                       rOUT[2])
                rp = batoid.CoordTransform(coordSysp,
                                        batoid.CoordSys(origin=np.array([0, 0, 0]), rot=np.identity(3))).applyForwardArray(rIN[0],
                                                                                                                        rIN[1],
                                                                                                                        rIN[2])
                wholeRayTracing[i] = np.array([rp, r]).transpose()
        return wholeRayTracing

    def visualized2D(nbOfRaysPerField, *args):
        colors = np.array([['indigo'], ['blue'], ['green'], ['yellow'], ['orange'], ['red']])
        c, i = 0, 0
        fig = px.line(width=1000, height=700)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                fig.add_trace(px.line(arg, x='z', y='y', color_discrete_sequence=['black'],).data[0])
            else:
                for j in range(0, len(arg)): # start at 1 => no ray from inf
                    if c > len(colors):
                        c = 0
                    line = arg[j]
                    df = pd.DataFrame({"x":line[0], "y":line[1], "z":line[2]})
                    if (i < nbOfRaysPerField) or (nbOfRaysPerField == 0):
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[0]).data[0]) #colors[c]
                    elif ((i >= nbOfRaysPerField) and (i < 2*nbOfRaysPerField)) and (nbOfRaysPerField != 0):
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[2]).data[0]) #colors[c]
                    else:
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[-1]).data[0]) #colors[c]
                c += 1
                i += 1
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')),
                          paper_bgcolor='rgb(255,255,255)',
                          plot_bgcolor='rgb(255,255,255)')
        fig.show()

    def visualized3D(nbOfRaysPerField, *args):
        colors = np.array([['indigo'], ['blue'], ['green'], ['yellow'], ['orange'], ['red']])
        c, i = 0, 0
        fig = px.line_3d(width=1000, height=700)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                fig.add_trace(px.line_3d(arg, x='x', y='y', z='z', color_discrete_sequence=['black'],).data[0])
            else:
                for j in range(0, len(arg)): # start at 1 => no ray from inf
                    if c > len(colors):
                        c = 0
                    line = arg[j]
                    df = pd.DataFrame({"x":line[0], "y":line[1], "z":line[2]})
                    if (i < nbOfRaysPerField) or (nbOfRaysPerField == 0):
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[0]).data[0]) #colors[c]
                    elif ((i >= nbOfRaysPerField) and (i < 2*nbOfRaysPerField)) and (nbOfRaysPerField != 0):
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[2]).data[0]) #colors[c]
                    else:
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[-1]).data[0]) #colors[c]
                c += 1
                i += 1
        camera_params = dict(up=dict(x=0,y=1,z=0),
                             center=dict(x=0,y=0,z=0),
                             eye=dict(x=-1.5,y=0,z=0.455))
    
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(zaxis=dict(title='z (meters)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')),
                          scene_camera=camera_params)
        fig.show()