# --Third Party--
import numpy as np
import xarray as xr
import numba as nb

class storage:
    # object for storing and expanding dictionary of arrays of similar shape from
    # multiple NetCDF files.
    def __init__(self, params):
        self.params = params
        self.merged = False

        self.storage = {}
        for p in self.params :
            self.storage[p]=[]
    
    def addFile(self, ds):
        # adding file to dictionary
        for p in self.params:
            self.storage[p].append(ds[p].values)
        self.merged = False
    
    def merge(self):
        # merge list of arrays to single np arrays
        for p in self.params:
            self.storage[p] = np.concatenate(self.storage[p])
        self.merged = True     


class n5esmrStorage(storage):
    def __init__(self, params):
        super().__init__(params) 

    def remove_outer_beams(self, firstbeam:int, lastbeam:int):
        # Removing outer beams of swath
        if not self.merged:
            raise 'Parameter arrays have not been merged'
        for p in self.params:
            self.storage[p] = self.storage[p][:,firstbeam:lastbeam]


# %% Satellite angles in degree
spos2ia=np.array([\
        -63.85785,  -61.69076,  -59.61496,  -57.61278,  -55.67122,\
        -53.78044,  -51.93276,  -50.12207,  -48.34344,  -46.59283,\
        -44.86687,  -43.16271,  -41.47795,  -39.81051,  -38.15862,\
        -36.52070,  -34.89540,  -33.28150,  -31.67793,  -30.08373,\
        -28.49804,  -26.92008,  -25.34914,  -23.78458,  -22.22581,\
        -20.67228,  -19.12349,  -17.57897,  -16.03828,  -14.50102,\
        -12.96678,  -11.43522,   -9.90597,   -8.37870,   -6.85308,\
        -5.32880,   -3.80556,   -2.28305,   -0.76098,    0.76095,\
        2.28302,    3.80553,    5.32877,    6.85305,    8.37867,\
        9.90594,   11.43519,   12.96676,   14.50099,   16.03825,\
        17.57894,   19.12346,   20.67225,   22.22578,   23.78455,\
        25.34911,   26.92005,   28.49801,   30.08370,   31.67790,\
        33.28147,   34.89537,   36.52067,   38.15859,   39.81048,\
        41.47792,   43.16268,   44.86684,   46.59280,   48.34341,\
        50.12204,   51.93272,   53.78041,   55.67119,   57.61274,\
        59.61492,   61.69072,   63.85781])



# %% RTM model
@nb.jit
def esmr(V,W,L,Ts,Ti,c_ice,theta):
    """---------------------------------------------------------------------------
    Wentz: a well calibrated ocean algorithm for ssm/i. JGR 102(C4), 8703-8718, 1997
    with modifications to work at incidence angles between nadir and 60 deg and over sea ice
    computes the brightness temperature of the ocean surface at 19 GHz horisontal polarisation
    Tb=f(V,W,L,Ts,Ti,c_ice,theta)
    V: columnar water vapor [mm]
    W: windspeed over water [m/s], 10m
    L: columnar cloud liquid water [mm]
    Ts: sea surface temperature [K, i.e. SST
    Ti: ice surface temperature [K], i.e. emitting layer temperature
    c_ice: ice concentration [0-1]
    theta: incidence angle [deg]
    -------------------------------------------------------------------------------"""

    thetar=0.0174533*theta
    #the ice emissivity as a function of incidence angle, based MEMLS simulations for 19h, avg. for winter months Nov-Mar
    x=[0.00, 5.00, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0]
    e=[0.9318, 0.9316, 0.9309, 0.9296, 0.9278, 0.9254, 0.9222, 0.9180, 0.9126, 0.9058, 0.8994, \
       0.8883, 0.8713, 0.8]
    #updated first-year ice emissivities
    x_fy=[0.0,      5.0,     10.0,     15.0,     20.0,     25.0,     30.0,\
         35.0,  40.0,        45.0,     50.0,     55.0,     60.0,     65.0]
    e_fy=[0.910430, 0.909988, 0.908647, 0.906359, 0.903037, 0.898555, 0.892729, \
          0.885306, 0.875927, 0.864084, 0.849024, 0.829596, 0.803955, 0.769277]
    #linear interpolation
    #fxe=interp1d(x,e)
    #e_ice=fxe(theta)
    e_ice = np.interp(theta,x,e)

    #table 1: atmospheric coefficients for 19h
    c0=240.0E+0
    c1=305.96E-2
    c2=-764.41E-4
    c3=885.95E-6
    c4=-40.80E-7
    c5=0.60E+0
    c6=-0.16E+0
    c7=-2.13E-2
    a0=11.80E+0
    av1=2.23E-3
    av2=0.00E-5

    #table 2: water surface scattering, reflected downwelling for 19h
    Xi=0.688


    #the isotropic Tb radiative transfer equation eq 15
    #Tb=TBU+tau*(emissivity*Ts+(1-emissivity)*(omega*TBD+tau*TBC))
    #eq 18a
    #if V <= 48.0: Tv=273.16+0.8337*V-3.029E-5*(V**3.33)
        #eq 18b
    #else:  Tv=301.16
    Tv=273.16+0.8337*V-3.029E-5*(V**3.33)
    #eq 22 + 23
    Tl=(Ts+273.0)/2.0
    Al37=0.208*(1-0.026*(Tl-283))*L
    Al=0.2858*Al37

    #cosmic background
    TBC=2.7

    #eq17a
    TD=c0+c1*V+c2*(V**2)+c3*(V**3)+c4*(V**4)+c5*(Ts-Tv)
    #eq17b
    TU=TD+c6+c7*V
    #eq 20
    A0=(a0/TD)**1.4
    #eq 21
    Av=av1*V+av2*(V**2)

    #eq 19 transmitance through the atmosphere
    tau=np.exp((-1.0/np.cos(thetar))*(A0+Av+Al))
    #eq 16a up-welling Tb
    TBU=TU*(1.0-tau)
    #eq 16b down-welling Tb
    TBD=TD*(1.0-tau)

#water permittivity as a function of temperature, Ulaby et al. 1986 E64a
    temp=[-100,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,\
          16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,\
          34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]

    permre=[15.3,15.3,15.9,16.5,17.1,17.8,18.5,19.2,20.0,20.7,21.5,\
           22.3,23.2,24.0,24.9,25.7,26.6,27.5,28.4,29.3,30.2,31.1,\
           31.9,32.8,33.6,34.5,35.3,36.1,36.8,37.6,38.3,39.0,\
           39.8,40.4,41.0,41.6,42.2,42.7,43.2,43.6,44.1,44.5,\
           44.9,45.3,45.7,46.0,46.3,46.7,47.0,47.3,47.6,47.9,\
           48.2,48.4,48.7,49.0,49.3]

    permim=[28.0,28.0,28.6,29.1,29.7,30.3,30.9,31.4,31.9,32.4,32.9,\
           33.4,33.8,34.2,34.6,35.0,35.3,35.6,35.9,36.1,36.3,36.4,\
           36.6,36.7,36.7,36.8,36.8,36.7,36.7,36.6,36.5,36.4,\
           36.3,36.1,36.0,35.8,35.6,35.4,35.2,34.9,34.7,34.5,\
           34.3,34.0,33.8,33.5,33.3,33.0,32.8,32.5,32.3,32.0,\
           31.7,31.4,31.1,30.8,30.5]
    

    #linear interpolation
    #real part of the water permittivity at 19.35GHz, 34psu and temperature
    e_w_r = np.interp(Ts-273.15,temp,permre)

    #imag part of the water permittivity at 19.35GHz, 34psu and temperature
    e_w_i = np.interp(Ts-273.15,temp,permim)

    #eq.1.52 in Schanda: physical fundamentals of remote sensing 1986.
    p=(1/np.sqrt(2))*(((e_w_r - np.sin(thetar)**2)**2 + e_w_i**2)**0.5 + (e_w_r - np.sin(thetar)**2))**0.5
    q=(1/np.sqrt(2))*(((e_w_r - np.sin(thetar)**2)**2 + e_w_i**2)**0.5 - (e_w_r - np.sin(thetar)**2))**0.5
    rih=((p - np.cos(thetar))**2 + q**2)/((p + np.cos(thetar))**2 + q**2)

    E0=(1.0-np.abs(rih))

    #Ulaby+ Meissner and Wentz 2012        
    Ew=(0.0094*theta+0.3)*W/Ts

    #eq 29 sea surface slope variance: sigma**2
    sigma=np.sqrt(5.22E-3*Xi*W)
    #if sigma**2 > 0.07: sigma=math.sqrt(0.07)
    #eq 28 reflection reduction factor due to surface roughness
    omega=1.0+6.1*(sigma**2-68.0*sigma**6)*tau**2
    #eq 24
    emissivity=E0+Ew
    #eq 16
    Tb=TBU+tau*((1.0-c_ice)*emissivity*Ts+c_ice*e_ice*Ti+\
                (1.0-c_ice)*(1.0-emissivity)*(omega*TBD+tau*TBC)\
                    +c_ice*(1.0-e_ice)*(TBD+tau*TBC))
    return Tb