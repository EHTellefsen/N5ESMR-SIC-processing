"""
Script for extracting running mean, standard deviation and number of none-NaN points in period.
Data is used for LDTP algorithm.

Author: Emil Haaber Tellefsen
Co-Author: Rasmus Tage Tonboe

Date: 04/11/2024
"""

# --Build in--
import os
import warnings

# --Proprietary--

# --Third Party--
import numpy as np
from glob import glob
from tqdm import tqdm
import xarray as xr

#ignore missing data warnings as setting values to NaN is intended behavior
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")

# %% constants
DATADIR = './outputs/N5ESMR_out'
R_SD_SAVEDIR = './outputs/LDTP/rSD'
R_MEAN_SAVEDIR = './outputs/LDTP/rMean'
R_NPOINTS_SAVEDIR = './outputs/LDTP/rNPoints'
HEMISPHERE = 'S'
FIRSTDATE = '1972-12-11'
LASTDATE = '1977-05-16'
PERIOD_LENGTH = 15
ARRAYSIZE = (432,432)

class periodicStorage:
    def __init__(self, ArraySize: tuple):
        self.N = ArraySize[0]
        self.M = ArraySize[1]
        self.dateRange = np.arange(np.datetime64(FIRSTDATE,'D'),np.datetime64(LASTDATE,'D'))
        self.data = np.empty((PERIOD_LENGTH,self.N,self.M))
        self.cdate = self.dateRange[0]

        #load first files into the current data array s.t. data for the initial prediction is used
        for i, date in enumerate(np.arange(self.cdate - PERIOD_LENGTH//2, self.cdate + PERIOD_LENGTH//2 + 1)):
            self.data[i] = self.loadFile(date)
        
    def loadFile(self, date):
        # function for loading Brightness temperature from nc file given consistent name pattern where date can be read
        date = str(date)
        year = date[:4]
        month = date[5:7]
        day = date[8:10]

        filename = ("ESACCI-SEAICE-L3C-SICONC-NIMBUS5_ESMR-EASE2_%sH-%s%s%s-fv1.1.nc" % (HEMISPHERE, year, month, day))
        file = glob(os.path.join(DATADIR, filename))

        # load Tb if file exists
        if file:
            ds = xr.load_dataset(file[0])
            Tb = ds['Tb_corr'].to_numpy()
            ds.close()
        
        # return array of NaNs in case no file exists
        else:
            Tb = np.ones((1, self.N, self.M))*np.nan 
        return Tb
    
    def calcAndSaveStats(self):
        # calculating running mean, standard deviation and number of non-NaN points
        rSD = np.nanstd(self.data,axis=0)
        rMean = np.nanmean(self.data,axis=0)
        rNPoints = np.sum(~np.isnan(self.data),axis=0)
        np.save(os.path.join(R_SD_SAVEDIR, 'rSD_'+HEMISPHERE +'H_'+str(self.cdate)+'.npy'),rSD)
        np.save(os.path.join(R_MEAN_SAVEDIR, 'rMEAN_'+HEMISPHERE +'H_'+str(self.cdate)+'.npy'),rMean)
        np.save(os.path.join(R_NPOINTS_SAVEDIR, 'rNPoints_'+HEMISPHERE +'H_'+str(self.cdate)+'.npy'),rNPoints)

    def next(self):
        # updating date and data array to reflect that
        self.cdate += 1
        self.data[:-1] = self.data[1:]
        self.data[-1] = self.loadFile(self.cdate + PERIOD_LENGTH//2)

    def run(self):
        # function for looping through full dateRange
        for date in tqdm(self.dateRange):
            self.calcAndSaveStats()
            self.next()

if __name__ == "__main__":
    ps = periodicStorage(ARRAYSIZE)
    ps.run()
