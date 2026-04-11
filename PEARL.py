import numpy as np
import time
from copy import deepcopy
import batoid
import matplotlib.pyplot as plt
from Batoid4LOFT import rotation, defineCoordinate, utils, optic, build, ray, details, correction, plot, layout, ZemaxFileParser

def STP(param=0):
    global WL, fieldBias
    global telescope, M1

    ### Extraction ###
    ZMX = ZemaxFileParser('STP_TMA_Mark_12F+M1_Bending_SBTest_5_HC01.zmx')
    ZMX_backdist, ZMX_thic12, ZMX_thic23, ZMX_thic34, ZMX_thic4d = ZMX.surface('2'), ZMX.surface('12'), ZMX.surface('21'), ZMX.surface('25'), ZMX.surface('30')
    ZMX_M1, ZMX_M2, ZMX_M3, ZMX_4 = ZMX.surface('M1'), ZMX.surface('M2'), ZMX.surface('M3'), ZMX.surface('M4-Fold Flat')
    ZMX_XP = ZMX.surface('Exit Pupil')
    ZMX_D, ZMX_D_coordbreak = ZMX.surface('Flat Detector'), ZMX.surface('31')
    ##################

    ### Primary ###
    R1, conic1, inner1, outter1 = 1/ZMX_M1.CURV/1000, ZMX_M1.CONI, 0, ZMX_M1.DIAM/1000
    M1coord = defineCoordinate().local()
    M1 = optic('M1', type='Quadric', RoC=R1, conic=conic1, inDiam=inner1, outDiam=outter1, coordSys=M1coord).mirror()
    #mirrorWithSpider(width=0.3, height=4) or mirror()
    ###############

    ### Seconday ###
    R2, conic2, outter2 = 1/ZMX_M2.CURV/1000, ZMX_M2.CONI, ZMX_M2.DIAM/1000
    pos2 = ZMX_thic12.DISZ/1000
    M2coord = defineCoordinate(posZ=pos2).local()
    M2 = optic('M2', type='Quadric', RoC=R2, conic=conic2, outDiam=outter2, coordSys=M2coord).mirror()
    ################

    ### Field Vane ###
    FVcoord = defineCoordinate(anglX=-1.7195040471007106, posY=0.19500789936941232, posX=param, posZ=-1.5175389428142736).local()
    FV = batoid.Baffle(batoid.Plane(),
                       obscuration=batoid.ObscNegation(batoid.ObscRectangle(width=2*0.13325533633157122,
                                                                            height=2*0.05130203937308275,
                                                                            x=0.0, y=0.0,
                                                                            theta=0.0)),
                       coordSys=FVcoord,
                       name="FV",
                       inDiam=0,
                       outDiam=0.13325533633157122)
    ##################

    ### Tertiary ###
    R3, conic3, outter3 = 1/ZMX_M3.CURV/1000, ZMX_M3.CONI, ZMX_M3.DIAM/1000
    pos3 = (ZMX_thic12.DISZ + ZMX_thic23.DISZ)/1000
    M3coord = defineCoordinate(posZ=pos3).local()
    M3 = optic('M3', type='Quadric', RoC=R3, conic=conic3, outDiam=outter3, coordSys=M3coord).mirror()
    ################

    ### Quaternary ###
    outter4 = ZMX_4.DIAM/1000
    pos4 = pos3 + ZMX_thic34.DISZ/1000
    M4coord = defineCoordinate(posZ=pos4).local()
    M4 = optic('M4', outDiam=outter4, coordSys=M4coord).flatMirror()
    ##################

    ### Exit Pupil ###
    outterXP = ZMX_XP.DIAM/1000
    XPcoord = defineCoordinate(posZ=pos4).local()
    XP = batoid.RefractiveInterface(name='XP',
                                    surface=batoid.Plane(),
                                    obscuration=batoid.ObscAnnulus(outterXP/2, outterXP/2+10),
                                    inDiam=0,
                                    outDiam=outterXP,
                                    inMedium=batoid.ConstMedium(1.0),
                                    outMedium=batoid.ConstMedium(1.0),
                                    coordSys=XPcoord,
                                    skip=False)
    ##################

    ### Detector ###
    outterD = ZMX_D.DIAM/1000
    posD = pos4 + ZMX_thic4d.DISZ/1000
    Dcoord = defineCoordinate(posY=ZMX_D_coordbreak.PARM2/1000, posZ=posD).local()
    D = optic('D', outDiam=outterD, coordSys=Dcoord).detector()
    ################

    ### Telescope ###
    telescope = build.compoundOptic(M1, M2, FV, M3, M4, XP, D, 
                                    pupilSize=ZMX.system_details.ENPD/1000, backDist=ZMX_backdist.DISZ/1000,
                                    EPcoord=defineCoordinate(posZ=0).local())

def showLayout(show='2D'):
    start = time.time()

    ## Number of rays per field
    nbOfRaysPerField = 6 # should be even

    ## Rays of central field
    anglX, anglY = 0, 0
    anglY += fieldBias
    field0 = layout.rayBundle(telescope, WL, M1.inDiam, M1.outDiam, anglX, anglY, nbOfRaysPerField)

    ## Rays of central field 1
    anglX, anglY = 0, -0.043
    anglY += fieldBias
    field1 = layout.rayBundle(telescope, WL, M1.inDiam, M1.outDiam, anglX, anglY, nbOfRaysPerField)

    ## Rays of central field 2
    anglX, anglY = 0, 0.043
    anglY += fieldBias
    field2 = layout.rayBundle(telescope, WL, M1.inDiam, M1.outDiam, anglX, anglY, nbOfRaysPerField)

    ## Optics' lines
    opticsContours2D = layout.getAllLines(telescope)
    opticsContours3D = deepcopy(opticsContours2D) ## important to make copy of the chain because it is nested 
                                                  ## (one use 'destroys' it)

    ## 2D visualization
    if show == '2D':
        layout.visualized2D(len(field0), *opticsContours2D,
                            *field0, *field1, *field2)
    
    ## 3D visualization
    if show == '3D':
        layout.visualized3D(len(field0), *opticsContours3D,
                            *field0, *field1, *field2)
    
    print('2D or 3D calculation + plot time', np.round(time.time()-start, decimals=5), 's')

def showSpotDiagram(fieldX=0, fieldY=0, pattern='Hexapolar', density=1, scale=None):      
    infoTelescope = details(telescope, fieldX=fieldX, fieldY=fieldY, wavelength=WL)
    airyR = infoTelescope.airyRadius(offFieldY=fieldBias)

    # print('Out/In Flux                  :', np.round(np.sum(rays.flux)/np.sum(rays0.flux)*100, decimals=2), '%')
    print('Working F/#                  :', np.round(infoTelescope.workingFNumber(offFieldY=fieldBias), decimals=5))

    plot.spotDiagram(telescope, fieldX=fieldX, fieldY=fieldY, wavelength=WL, fieldBias=fieldBias, density=density, pattern=pattern, airyR=airyR, scale=scale)

def showFootPrint(surface=None, fieldX=0, fieldY=0, pattern='Hexapolar', density=1, centered=True):
    infoTelescope = details(telescope, fieldX=fieldX, fieldY=fieldY, wavelength=WL)

    FP = infoTelescope.footPrint(surface=surface, pattern=pattern, density=density, globalCoord=True)

    plot.footPrint(FP, centered=centered)

def showWavefrontMap(fieldX=0, fieldY=0, npx=1024):
    infoTelescope = details(telescope, fieldX=fieldX, fieldY=fieldY, wavelength=WL)
    WF = infoTelescope.wavefront(npx=npx, background=False)

    XPdiam = utils.XPdiam(telescope, theta_x=fieldX, theta_y=fieldY, wavelength=WL)
    XPpos = utils.exitPupilPos(optic=telescope, wavelength=WL)
    print(XPdiam*1000, (XPpos[-1] - telescope.items[-1].coordSys.origin[-1])*1000)

    plot.wavefront(WF, WL, [fieldX, fieldY-fieldBias], title='batoid', rotation='180')

    print('\n')
    infoTelescope.wavefrontZnk(npx=npx, jmax=15, eps=0, show=True)

    xp, yp = utils.getXY(telescope, fieldX, fieldY, WL, npx)
    Znk = infoTelescope.wavefrontZnk(npx=npx, jmax=37, eps=0, show=False)
    WF, removed = correction.removeTiltWavefront(WF, Znk, xp, yp, removeTilt=True, removePiston=False)

    plot.wavefront(WF, WL, [fieldX, fieldY-fieldBias], title='batoid removed tilt', rotation='180', tiltrmv=removed)

def main():
    #%run Batoid4LOFT.ipynb

    plt.close('all')

    ### Field ###
    global fieldBias
    fieldBias = 0.17
    fieldX, fieldY = 5/60, 2/60+fieldBias
    #############

    ### WaveL ###
    global WL
    WL = 650e-9
    #############

    STP()
    # showSpotDiagram(pattern='Dithered', density=5, fieldX=fieldX, fieldY=fieldY, scale=40) # Dithered - Hexapolar - Uniform
    showFootPrint(surface='XP', pattern='Uniform', density=100, fieldX=fieldX, fieldY=fieldY, centered=False) # Dithered - Hexapolar - Uniform
    # showLayout(show='2D')
    # showWavefrontMap(fieldX=fieldX, fieldY=fieldY, npx=1024)

if __name__=='__main__':
    # pass
    
    main()