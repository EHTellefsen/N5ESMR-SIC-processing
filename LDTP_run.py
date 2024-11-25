"""
Operation code for post-processing of corrected daily Sea Ice Concentration (SIC) for the 
Nimbus 5 Electrically Scanning Microwave Radiometer (N5ESMR). Method uses previous tie-points as
baseline and develops a local dynamical tiepoint framework which updates tiepoints from the signal 
stability of the same pixel

Author: Emil Haaber Tellefsen
Co-Author: Rasmus Tage Tonboe

Date: 11/11/2024
"""

# --Build in--
import os
import sys
import json
import warnings

# --Proprietary--
sys.path.append('./functions')
from makeNetCDF import LDTP_NetCDF

# --Third Party--
import numpy as np
from glob import glob
from tqdm import tqdm
import xarray as xr
from scipy import ndimage
import pyresample as pr

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# %% constants
DATADIR = './outputs/N5ESMR_out'
R_SD_DIR = './outputs/LDTP/rSD'
R_MEAN_DIR = './outputs/LDTP/rMean'
R_NPOINTS_DIR = './outputs/LDTP/rNPoints'
MASKDIR = './reference_files'
TPDIR = './outputs/newTP15'
SAVEDIR = './outputs/LDTP/LDTP_out'
ITPSAVEDIR = './tests_etc/ITP'
AREAFILE = './reference_files/areas_ease_v2atle.cfg'
HEMISPHERE = 'S'
FIRSTDATE = '1972-12-11'
LASTDATE = '1977-05-16'

MIN_POINTS = 7
R_SD_THRESH = 3.737038
FORGET_TIME = 180
ITP_LIMITS = (205.0, 255.0)
ARRAYSIZE = (432,432)


def smearingError(img: np.ndarray,
                  kernelsize: int):
    (n,m) = img.shape
    dst = np.empty((n,m))
    edge = kernelsize//2

    for i in range(n):
        for j in range(m):
            SW = img[max(0,i-edge):min(n,i+edge), max(0,j-edge):min(m,j+edge)]
            dst[i,j] = np.nanmax(SW) - np.nanmin(SW)
    return dst


class LDTP:
    def __init__(self):
        
        # Dates
        self.dateRange = np.arange(np.datetime64(FIRSTDATE,'D'),np.datetime64(LASTDATE,'D'))

        # Finding relevant data files
        self.TPFiles = glob(TPDIR + '/*')

        # Loading default tiepoints
        if HEMISPHERE == 'N':
            ITP_name = 'arctic_ice'
            OTP_name = 'arctic_ocean'
            self.maskfile = os.path.join(MASKDIR, 'LandOceanLakeMask_nh_ease2-250_v2.nc')
            self.area_def = pr.parse_area_file(AREAFILE, 'ease_nh')[0]
        elif HEMISPHERE == 'S':
            ITP_name = 'antarctic_ice'
            OTP_name = 'antarctic_ocean'
            self.maskfile = os.path.join(MASKDIR, 'LandOceanLakeMask_sh_ease2-250_v2.nc')
            self.area_def = pr.parse_area_file(AREAFILE, 'ease_sh')[0]
        
        self.refTP = {'ITP': np.empty(len(self.dateRange)), 
                           'ITP_std': np.empty(len(self.dateRange)),
                           'OTP': np.empty(len(self.dateRange)),
                           'OTP_std': np.empty(len(self.dateRange))}

        for i, date in enumerate(self.dateRange):
            self.refTP['ITP'][i], self.refTP['OTP'][i], self.refTP['ITP_std'][i], self.refTP['OTP_std'][i] = self.loadTiePointFile(date, ITP_name, OTP_name)

        # Initialize LDTP Array for ice tie points and daysSinceLastUpdate array for remembering when last updated
        self.LDTP = np.ones(ARRAYSIZE)*self.refTP['ITP'][0]
        self.LDTP_std = np.ones(ARRAYSIZE)*self.refTP['ITP_std'][0]

        self.daysSinceLastUpdate = np.zeros(ARRAYSIZE,dtype=int)
        self.uninitialized = np.ones(ARRAYSIZE,dtype=bool)



    def freeRun(self, backwards = False):   
        # inverting order of dates in case of reverse run
        if backwards:
            idx = np.flip(np.arange(len(self.dateRange)))
        else:
            idx = np.arange(len(self.dateRange))

        # looping over dates and updating LDTP
        for i in tqdm(idx):
            self.updateLDTP(i)

    
    def mainRun(self, TPsafeDir = None):
        
        idx = np.arange(len(self.dateRange))
        for i in tqdm(idx):
            
            # specify current date
            date_s = str(self.dateRange[i])
            year = date_s[:4]
            month = date_s[5:7]
            day = date_s[8:10]               
            
            self.updateLDTP(i)

            if TPsafeDir is not None:
                np.save(os.path.join(TPsafeDir, 'LDTP_'+HEMISPHERE +'H_'+date_s+'.npy'),self.LDTP)
            
            # loading Tb data
            dataFile = os.path.join(DATADIR,"ESACCI-SEAICE-L3C-SICONC-NIMBUS5_ESMR-EASE2_" + HEMISPHERE + "H-"+year+month+day+"-fv1.1.nc")
            if os.path.exists(dataFile):
                ds = xr.open_dataset(dataFile)
                Tb = ds['Tb_corr'].to_numpy()[0]
                Tb_raw = ds['Tb_corr'].to_numpy()[0]
                oldFlags = ds['status_flag'].to_numpy()[0]
                ds.close()

                # calculating sea ice concentration
                dTP = self.LDTP - self.refTP['OTP'][i]
                SIC = (Tb - self.refTP['OTP'][i])/dTP
                SIC_clip = np.clip(SIC, 0.0, 1.0)

                # quantify uncertainties
                smearing_uncertainty = smearingError(SIC_clip,3)
                delta_Tb_water = (-(1 - SIC_clip)*self.refTP['OTP_std'][i] / dTP)**2
                delta_Tb_ice = (-SIC_clip*self.LDTP_std / dTP)**2
                algorithm_uncertainty = np.sqrt(delta_Tb_water + delta_Tb_ice)
                total_uncertainty = np.sqrt(smearing_uncertainty**2 + algorithm_uncertainty**2)

                # making flags and filter
                flags, SIC_filtered, total_uncertainty_filtered = self.flagsAndCorrections(SIC, total_uncertainty, oldFlags, month)

                # creating data vars for output
                data_vars = {'raw_ice_conc_values': SIC, 
                             'algorithm_standard_error': algorithm_uncertainty,
                             'Tb_corr': Tb,
                             'Tb': Tb_raw,
                             'smearing_standard_error': smearing_uncertainty,
                             'total_standard_error': total_uncertainty_filtered,
                             'status_flag': flags,
                             'ice_conc': SIC_filtered}

                # making output NetCDF
                LDTP_NetCDF(data_vars, date_s, HEMISPHERE, self.area_def, SAVEDIR)
            i+=1



    # %% Helpers
    def loadTiePointFile(self, date, ITP_name, OTP_name):
        date = str(date)
        year = date[:4]
        month = date[5:7]
        day = date[8:10]

        # locating file
        filename = "Nimbus5-ESMR_%sm%s%s.json" % (year, month, day)
        file = os.path.join(TPDIR, filename)
        if os.path.exists(file):
            TP_data = json.load(open(file))
            OTP = TP_data[OTP_name]['mean_Tb_corr']
            OTP_std = TP_data[OTP_name]['std_Tb_corr']
            ITP = TP_data[ITP_name]['mean_Tb_corr']
            ITP_std = TP_data[ITP_name]['std_Tb_corr']
        
        # adding nan if file does not exist (given script this should never happen)
        else:
            OTP = np.nan
            OTP_std = np.nan
            ITP = np.nan
            ITP_std = np.nan
        
        return ITP, OTP, ITP_std, OTP_std
    

    def updateLDTP(self, i):
        date_s = str(self.dateRange[i])

        #loading data files
        rMean = np.load(os.path.join(R_MEAN_DIR, 'rMEAN_' + HEMISPHERE+'H_' + date_s + '.npy'))
        rSD = np.load(os.path.join(R_SD_DIR, 'rSD_' + HEMISPHERE+'H_' + date_s + '.npy'))
        rNPoints = np.load(os.path.join(R_NPOINTS_DIR, 'rNPoints_' + HEMISPHERE+'H_' + date_s + '.npy'))

        # checking if points should be updated
        toBeUpdated = (rNPoints >= MIN_POINTS) & (rSD < R_SD_THRESH) & (rMean > ITP_LIMITS[0]) & (rMean < ITP_LIMITS[1])

        # updating points
        self.LDTP[toBeUpdated] = rMean[toBeUpdated]
        self.LDTP_std[toBeUpdated] = rSD[toBeUpdated]

        # initializing/uninitializing pixels
        self.uninitialized[toBeUpdated] = False
        self.uninitialized[self.daysSinceLastUpdate>FORGET_TIME] = True

        # updating LDTP
        self.LDTP[self.uninitialized] = self.refTP['ITP'][i]
        self.LDTP_std[self.uninitialized] = self.refTP['ITP_std'][i]

        # updating when last update took place
        self.daysSinceLastUpdate[toBeUpdated] = 0
        self.daysSinceLastUpdate[~toBeUpdated] += 1
    

    def flagsAndCorrections(self, SIC, SIC_std, oldFlags, month):
        SIC_filtered = np.copy(SIC)
        SIC_std_filtered = np.copy(SIC_std)
        
        # load masks
        maskdata = xr.open_dataset(self.maskfile)
        smask = maskdata['smask_sicci'].to_numpy().astype(int)
        climatology = maskdata['climatology'][int(month) - 1].to_numpy()
        climatology[np.isnan(climatology)]=-1
        climatology = climatology.astype(int)
        maskdata.close()
        
        #updating flags
        flag = np.ones(oldFlags.shape)*4
        flag[oldFlags==16]=16
        
        # climatology flag (64)
        flag[climatology == 0] = 64
        SIC_filtered[climatology == 0] = 0.0

        # land (1) /coast (2) /lake flags (32)
        flag[smask == 2]=1 # land
        flag[smask >= 4]=2 # lake
        flag[smask == 1]=32 # coast
        invalid_surface = smask>0
        
        # land-spill-over flag (8)
        smask = smask.astype(float)
        smask[smask > 0.5]=1
        smask*=0.9
        smaskb = ndimage.uniform_filter(smask,size=5)
        smaskb[smask > 0.5]=0.9
        smaskb[smaskb < 0.01]=-1.0
        land_spill_over = smaskb - smask
        flag[land_spill_over > SIC_filtered] = 8 #land-spill-over correction, flag 8
        SIC_filtered[land_spill_over > SIC_filtered] = 0.0

        # Apply surface mask
        SIC_filtered[invalid_surface] = np.nan
        SIC_std_filtered[invalid_surface] = np.nan

        # Invalid point flag (128)
        flag[(flag==4) & np.isnan(SIC_filtered)] = 128
        flag[(flag==4) & (SIC_filtered>=0)] = 0

        #clipping SIC
        SIC_filtered = np.clip(SIC_filtered,0,1)

        return flag, SIC_filtered, SIC_std_filtered



# %% main
if __name__ == "__main__":
    ldtp = LDTP()
    ldtp.freeRun()
    ldtp.freeRun(backwards=True)
    ldtp.mainRun(ITPSAVEDIR)
