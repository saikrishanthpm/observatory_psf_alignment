import batoid
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from datetime import date

from .details import details
from .utils import utils
from .ray import ray
from .definecoordinate import defineCoordinate
from .optic import optic
from .build import build

def is_running_in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            return True
    except ImportError:
        pass
    return False

if not is_running_in_notebook():
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import sys
    import matplotlib
    matplotlib.use('TkAgg')

def create_window(fig, output_dict):
    """Creates a GUI window with a matplotlib plot on the left and print outputs on the right."""
    # If there is no existing Tk instance, create one
    if not hasattr(create_window, "_root"):
        create_window._root = tk.Tk()
        create_window._root.withdraw()  # Hide the root window
        create_window._active_windows = 0  # Track active windows

    # Increment the active window counter
    create_window._active_windows += 1

    # Create a new Toplevel window
    window = tk.Toplevel(create_window._root)
    window.title("Viewer")

    # Configure layout
    window.rowconfigure(0, weight=1)  # Plot row
    window.rowconfigure(1, weight=0)  # Output row (doesn't expand)
    window.columnconfigure(0, weight=1)  # Single column layout

    # Plot frame (top)
    plot_frame = tk.Frame(window, bg="white", bd=0, highlightthickness=0)
    plot_frame.grid(row=0, column=0, sticky="nsew")

    # Output frame (bottom)
    output_frame = tk.Frame(window, bg="white", bd=0, highlightthickness=0)
    output_frame.grid(row=1, column=0, sticky="nsew")

    # Toolbar (above the plot)
    toolbar_frame = tk.Frame(plot_frame, bg="white", bd=0, highlightthickness=0)
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)

    # Add matplotlib figure to plot frame
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add toolbar for matplotlib navigation (inside toolbar_frame)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    # Add text widget for output
    text_widget = tk.Text(output_frame, wrap=tk.WORD, state=tk.NORMAL, bg="white", fg="black", height=6)
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Insert output dictionary as lines
    for key, value in output_dict.items():
        text_widget.insert(tk.END, f"{value}\n")
    text_widget.configure(state=tk.DISABLED)

    # Quit button to close only this window
    def on_quit():
        nonlocal window
        window.destroy()  # Destroy this Toplevel window
        create_window._active_windows -= 1  # Decrement active window counter

        # Exit the main loop if no more windows are open
        if create_window._active_windows == 0:
            create_window._root.quit()

    # Ensure the window remains open until explicitly closed
    window.protocol("WM_DELETE_WINDOW", on_quit)


class plot:

    def sag3D(optic):
        inDiam, outDiam = optic.inDiam, optic.outDiam
        th = 2 * np.pi * np.random.rand(int(1e6))
        cth, sth = np.cos(th), np.sin(th)
        d = (outDiam - inDiam) * np.random.rand(int(1e6)) + inDiam
        x = (d/2) * cth
        y = (d/2) * sth
        z = optic.surface.sag(x, y) * 1000
        
        xi = np.linspace(x.min(), x.max(), 64)
        yi = np.linspace(y.min(), y.max(), 64)
        X,Y = np.meshgrid(xi, yi)
        Z = griddata((x,y), z, (X,Y), method='nearest')
        
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (mm)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(range=[np.min(Z), 0],title='sag (mm)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()

    def sag3D_TEST(surface, outDiam):
        inDiam = 0
        th = 2 * np.pi * np.random.rand(int(1e6))
        cth, sth = np.cos(th), np.sin(th)
        d = (outDiam - inDiam) * np.random.rand(int(1e6)) + inDiam
        x = (d/2) * cth
        y = (d/2) * sth
        z = surface.sag(x, y)
        
        xi = np.linspace(x.min(), x.max(), 64)
        yi = np.linspace(y.min(), y.max(), 64)
        X,Y = np.meshgrid(xi, yi)
        Z = griddata((x,y), z, (X,Y), method='nearest')
        
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (waves)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(title='sag (waves)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()
        return xi, yi, Z     
    def sag3D_TEST_residual(xi, yi, Z):  
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (waves)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(title='residual sag (waves)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()

    def spotDiagram(optic, fieldX=0, fieldY=0, wavelength=633e-9, fieldBias=0, density=1, pattern='Hexapolar', airyR=None, scale=None, source=None):
        rvImg = details(optic=optic, fieldX=fieldX, fieldY=fieldY, wavelength=wavelength).spotDiagram(pattern=pattern, density=density, source=source)
        rvImg = rvImg.toCoordSys(batoid.globalCoordSys)
        
        ## Centroid method
        cx = np.sum(rvImg.x)/len(rvImg.x)
        cy = np.sum(rvImg.y)/len(rvImg.y)

        anglX, anglY = np.round(fieldX, decimals=3), np.round(fieldY-fieldBias, decimals=3)

        fig, ax = plt.subplots()
        ax.scatter((rvImg.x - cx) * 1e6, (rvImg.y - cy) * 1e6, s=0.25 ,c='blue', marker='.') # s=0.25 marker='+'
        
        if airyR is not None:
            ax.add_patch(plt.Circle((0, 0), airyR, color='k', fill=False))
        ax.set_title(f'OBJ: {anglX}, {anglY} (deg)', fontsize=14)
        ax.set_xlabel(f'IMA: {np.round(cx * 1000, decimals=3)}, {np.round(cy * 1000, decimals=3)} mm', fontsize=14)
        if scale is not None:
            ax.set_xlim(-scale/2, scale/2)
            ax.set_ylim(-scale/2, scale/2)
            ax.set_ylabel(f'{float(scale)}', fontsize=14)
        else:
            lim = ax.get_xlim()
            ax.set_ylabel(f'{float(lim[1]-lim[0]):.1f}', fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both',       # changes apply to the both axis
                       which='both',      # both major and minor ticks are affected
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labelleft=False)
        ax.grid()
        ax.set_axisbelow(True)
        fig.canvas.header_visible = False
        plt.subplots_adjust(left=-0.1)

        print_dict = {}

        print_dict['Date'] = f'{date.today().strftime("%m/%d/%Y")}'
        if airyR is not None:
            print_dict['Airy Radius'] = f'Units are µm. Airy Radius: {np.round(airyR, decimals=3)} µm. Wavelength: {wavelength * 1e6} µm.'
        Rms_R = np.sqrt(np.sum((rvImg.x - cx)**2)/len(rvImg.x) + np.sum((rvImg.y - cy)**2)/len(rvImg.y)) * 1e6
        print_dict['RMS radius'] = f'RMS radius :    {np.round(Rms_R, decimals=3)}'
        R = np.sqrt((rvImg.x - cx)**2 + (rvImg.y - cy)**2)
        Geo_R = np.max(R) * 1e6
        print_dict['GEO radius'] = f'GEO radius :    {np.round(Geo_R, decimals=3)}'
        if scale is not None:
            print_dict['Scale bar'] = f'Scale bar  :   {float(scale)}    Reference : Centroid\n'
        
        if is_running_in_notebook():
            plt.show()
            for key, value in print_dict.items():
                print(f"{value}")
        else:
            create_window(fig, print_dict)
            create_window._root.mainloop()

    def footPrint(rvImg, centered=True):
        ## Centroid method
        cx = np.sum(rvImg.x)/len(rvImg.x)
        cy = np.sum(rvImg.y)/len(rvImg.y)

        print_dict = {}

        ## Ellipse
        if centered:
            points = np.array([rvImg.x - cx, rvImg.y - cy])
        else:
            points = np.array([rvImg.x, rvImg.y])
        cov_matrix = np.cov(points)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        axis_ratio = np.sqrt(min(eigenvalues) / max(eigenvalues))
        ellipticity = 1 - axis_ratio
        width = np.sqrt(max(eigenvalues)) * 4
        height = np.sqrt(min(eigenvalues)) * 4
        mean = np.mean(points, axis=1)
        angle = -np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0]))
        print_dict['Axis Ratio'] = f"Axis Ratio: {axis_ratio}"
        print_dict['Ellipticity'] = f"Ellipticity: {ellipticity}"
        print_dict['Angle'] = f"Angle: {angle}"
        print_dict['Mean'] = f"Mean: {mean}"
        print_dict['WxH'] = f"Width: {width}, Height: {height}"

        fig, ax = plt.subplots()

        ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='red', facecolor='none')
        plt.gca().add_patch(ell)

        if centered:
            R = np.sqrt((rvImg.x - cx)**2 + (rvImg.y - cy)**2)
            plotR = R
            ax.scatter((rvImg.x - cx), (rvImg.y - cy), s=0.25 ,c='blue', marker='.')
        else:
            plotR = np.sqrt((rvImg.x)**2 + (rvImg.y)**2)
            R = np.sqrt((rvImg.x - cx)**2 + (rvImg.y - cy)**2)
            ax.scatter(rvImg.x, rvImg.y, s=0.25 ,c='blue', marker='.')
        Geo_R_plot = np.max(plotR)
        Geo_R = np.max(R)
        margin = 5/100
        
        ax.set_xlabel(f'CENTROID: {np.round(cx * 1000, decimals=3)}, {np.round(cy * 1000, decimals=3)} mm', fontsize=14)

        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both',       # changes apply to the both axis
                       which='both',      # both major and minor ticks are affected
                       bottom=False,
                       top=True,
                       left=False,
                       right=True,
                       labelbottom=False,
                       labelleft=False,
                       labelright=True,
                       labeltop=True)
        ax.set_xlim(-Geo_R_plot*(1+margin), Geo_R_plot*(1+margin))
        ax.set_ylim(-Geo_R_plot*(1+margin), Geo_R_plot*(1+margin))
        ax.grid()
        ax.set_axisbelow(True)
        fig.canvas.header_visible = False
        plt.subplots_adjust(left=-0.1)

        print_dict['Date'] = f'{date.today().strftime("%m/%d/%Y")}'
        print_dict['Radius'] = f'Radius :    {np.round(Geo_R, decimals=3)} m'

        if is_running_in_notebook():
            plt.show()
            for key, value in print_dict.items():
                print(f"{value}")
        else:
            create_window(fig, print_dict)
            create_window._root.mainloop()

    def wavefront(WF, wl, fields, title='', rotation='0', XPdiam=0, tiltrmv=None):
        if isinstance(WF, batoid.lattice.Lattice):
            WF = WF.array
        else:
            pass

        if rotation == '90':
            WF = np.rot90(WF)
        elif rotation == '180':
            WF = np.flipud(WF)
        elif rotation == '270':
            WF = np.flipud(WF)
            WF = np.rot90(WF)
        else:
            pass
        
        fig, ax = plt.subplots()
        WFax = ax.imshow(WF, cmap=plot.customDivergingColormap(), extent=[-1., 1., -1., 1.])
        ax.set_facecolor('#2822bb')
        #ax.set_title('Wavefront Map', fontsize=14)
        ax.set_xlabel('X-Pupil (Rel. Units)', fontsize=14)
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_ylabel('Y-Pupil (Rel. Units)', fontsize=14)
        ax.set_yticks([-1.0, 0.0, 1.0])
        cbar = fig.colorbar(WFax, label='waves')
        cbar.set_ticks(np.round(np.linspace(WF.min(), WF.max(), 11), decimals=4))
        #WFax.set_clim(-5, 5)
        #cbar.set_ticks(np.round(np.linspace(-5, 5, 11), decimals=4))
        fig.canvas.header_visible = False
        ax.set_title(title)

        print_dict = {}

        print_dict['Date'] = f'{date.today().strftime("%m/%d/%Y")}'
        print_dict['WL&field'] = f'{float(wl * 1e6)} µm at {np.round(float(fields[0]), decimals=4)}, {np.round(float(fields[1]), decimals=4)} (deg)'
        degsymb = '\u00b0'
        print_dict['Rotation'] = f'Rotation: {rotation}{degsymb}'
        RMS = WF.std()
        PV = np.abs(WF.max()-WF.min())
        print_dict['PV'] = f'Peak to valley = {np.round(PV, decimals=4)}, waves, RMS = {np.round(RMS, decimals=4)} waves'
        if XPdiam != 0:
            print_dict['EXD'] = f'Exit Pupil Diameter: {XPdiam * 1000} Millimeters'
        if tiltrmv is not None:
            if len(tiltrmv) == 0:
                pass
            elif len(tiltrmv) == 1:
                print_dict['Piston'] = f'Piston Removed : {np.round(tiltrmv[0], decimals=4)} waves'
            elif len(tiltrmv) == 3:
                print_dict['Pist&Tilts'] = f'Piston Removed : {np.round(tiltrmv[0], decimals=4)} waves, Tilt Removed:  X = {np.round(tiltrmv[1], decimals=4)}, Y = {np.round(tiltrmv[2], decimals=4)} waves\n'
            else:
                print_dict['Tiltsq'] = f'Tilt Removed:  X = {np.round(tiltrmv[0], decimals=4)}, Y = {np.round(tiltrmv[1], decimals=4)} waves\n'
        
        if is_running_in_notebook():
            plt.show()
            for key, value in print_dict.items():
                print(f"{value}")
        else:
            create_window(fig, print_dict)
            create_window._root.mainloop()
            

    def customDivergingColormap():
        nbOfBits = 512
        N = nbOfBits//7
        s = 1.2
        #jet_colors = plt.cm.get_cmap('jet')
        jet_colors = mpl.colormaps.get_cmap('jet')
        lower_colors = jet_colors(np.linspace(0.10, 1/3, int(s*N)))
        upper_colors = jet_colors(np.linspace(2/3, 0.95, int(1.5*s*N)))
        middle_colors = jet_colors(np.linspace(1/3, 2/3, int(nbOfBits-2.5*s*N)))
        custom_colors = np.vstack((lower_colors, middle_colors, upper_colors))
        return mcolors.ListedColormap(custom_colors)

    def findPxSize(PSF, airyR):
        N = len(PSF[0])
        mid = N//2
        CS = PSF[mid, :]
        minima = argrelextrema(CS, np.less)[0]
        indx = np.searchsorted(minima, mid)
        indx = minima[indx]
        pxSize = airyR/(indx-mid+1)
        return pxSize
        

    def psf3D(PSF, airyR, thresh=0):
        #PSF = np.log(PSF.array)
        #plot.thresholding(PSF, thresh=thresh)
        N = len(PSF[0])
        mid = N//2
        pxSize = plot.findPxSize(PSF, airyR)
        x = np.linspace(-mid*pxSize, mid*pxSize, N)
        x, y = np.meshgrid(x, x)
        
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=PSF,
                                         colorscale='Spectral',
                                         reversescale=True,
                                         colorbar=dict(thickness=30,
                                                       tickvals=np.round(np.linspace(PSF.min(), PSF.max(), 10), decimals=1)
                                                      )
                                        )
                             ]
                       )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          width=700,
                          height=500,
                          xaxis_range=[-airyR*6.557, airyR*6.557],
                          yaxis_range=[-airyR*6.557, airyR*6.557]
                         )
        fig.show()

    def _forward(x):
        return 10**x
    def _inverse(x):
        return np.log(x)
    
    def psf2D(PSF, airyR, thresh=0):
        #PSF = np.log(PSF)
        #plot.thresholding(PSF, thresh=thresh)
        N = len(PSF[0])
        mid = N//2
        pxSize = plot.findPxSize(PSF, airyR)
        
        fig, ax = plt.subplots()
        PSFax = ax.imshow(PSF, cmap='gray', norm=mcolors.LogNorm(vmin=1e-5, vmax=1), extent=[-mid*pxSize,mid*pxSize,-mid*pxSize,mid*pxSize])
        ax.set_title(f'PSF', fontsize=14)
        ax.set_xlabel('µm', fontsize=14)
        ax.set_ylabel('µm', fontsize=14)
        ax.set_xlim(-airyR*6.557, airyR*6.557)
        ax.set_ylim(-airyR*6.557, airyR*6.557)
        cbar = fig.colorbar(PSFax)
        cbar.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        fig.canvas.header_visible = False
        plt.show()

    def psfCrossSection(PSF, airyR, wl=None, fields=None, stehlRatio=None, axis='Y'):
        #PSF = np.sqrt(PSF)

        N = len(PSF[0])
        mid = N//2
        
        fig, ax = plt.subplots()
        if axis.casefold() == 'y':
            CS = PSF[:, mid]
            ax.set_title(f'Cross Section of PSF at Y=0', fontsize=14)

            pxSize = plot.findPxSize(PSF, airyR)
        elif axis.casefold() == 'x':
            CS = PSF[mid, :]
            ax.set_title(f'Cross Section of PSF at X=0', fontsize=14)

            pxSize = plot.findPxSize(PSF, airyR)

        x = np.linspace(-mid*pxSize, mid*pxSize, N)
        x -= x[CS.tolist().index(np.max(CS))]
        ax.plot(x, CS)
        ax.set_xlabel('µm', fontsize=14)
        ax.set_ylabel('Relative Irradiance', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xlim(-airyR*6.557, airyR*6.557)
        ax.set_yticks(np.arange(0, 1, 0.1))
        fig.canvas.header_visible = False
        plt.grid(which='major')
        plt.grid(which='minor', color='#EEEEEE', linestyle=':')
        ax.minorticks_on()
        plt.show()

        print(f'{date.today().strftime("%m/%d/%Y")}')
        if (wl is not None) and (fields is not None):
            print(f'{float(wl * 1e6)} µm at {float(fields[0])}, {float(fields[1])} (deg)')
        print(f'Image width is {2*airyR*6.557:.2f} µm')
        if stehlRatio is not None:
            print(f'Strehl ratio: {stehlRatio:.3f}')

    def thresholding(data, thresh=0):
        if thresh != 0:
            lim = thresh * np.max(data)
            data[data<lim] = 0
        return data

    def rayFan(opticalSys, fieldX=0, fieldY=0, wavelength=633e-9, axis='y', show=True):
        ## Incoming ray
        nbOfRays = 1e3
        rays0 = ray(opticalSys, fieldX=fieldX, fieldY=fieldY, wavelength=wavelength, nbOfRays=nbOfRays).fromPupilForFan(axis=axis)
        rays = rays0.copy()
        
        opticalSys.trace(rays, reverse=False)
        rays = ray.deleteVignettedRays(rays)

        ## Coordinates on Detector
        if axis == 'y':
            c = np.sum(rays.y)/len(rays.y)
            e = (rays.y - c) * 1e3
        elif axis == 'x':
            c = np.sum(rays.x)/len(rays.x)
            e = (rays.x - c) * 1e3

        ## Detector at Exit Pupil
        XPpos = utils.exitPupilPos(optic=opticalSys, wavelength=wavelength)[-1]
        Dcoord = defineCoordinate(posZ=XPpos).local()
        originalD = opticalSys.items[-1]
        D = optic('D', inDiam=originalD.inDiam, outDiam=originalD.outDiam, coordSys=Dcoord).detector()
        newSysAtXP = build.compoundOptic(*opticalSys.items[0:-1], D,
                                         pupilSize=opticalSys.pupilSize, backDist=opticalSys.backDist,
                                         EPcoord=opticalSys.stopSurface.coordSys)

        newSysAtXP.trace(rays0, reverse=False)
        rays0 = ray.deleteVignettedRays(rays0)

        ## Coordinates in Exit Pupil
        if axis == 'y':
            P = rays0.y / np.max(np.abs(rays0.y))
        elif axis == 'x':
            P = rays0.x / np.max(np.abs(rays0.x))
        
        ## Plot P=f(e)
        if show:
            fig, ax = plt.subplots()
            ax.plot(P, e, linestyle="-", marker=".", markersize=5, zorder=2)
            ybound = ax.get_ylim()
            ax.plot([0, 0], [ybound[0], ybound[1]], color='k', zorder=1)
            ax.plot([-1, 1], [0, 0], color='k', zorder=0)
            ax.grid()
            ax.set_xlim(-1, 1)
            ax.set_ylim(ybound[0], ybound[1])
            #plt.ylim(-0.01, 0.01)
            ax.set_title(f'Ray Fan on {axis.upper()}-axis for field of {fieldX}°, {fieldY}°')
            ax.set_ylabel(f'e{axis} (mm)')
            ax.set_xlabel(f'P{axis} (normalized)')
            plt.subplots_adjust(left=+0.17)
            plt.show()
        return e, P
