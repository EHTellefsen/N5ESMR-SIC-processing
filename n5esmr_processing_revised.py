# --Build in--
import os
import json
import sys

# --Proprietary--
sys.path.append('./functions')
import n5esmr
from makeNetCDF import makeNetCDF

# --Third Party--
import numpy as np
from glob import glob
from tqdm import tqdm
import pyresample as pr
from scipy import ndimage
import xarray as xr

# %% constants
DATADIR = 'D:/N5ESMR/ESMR_ERA5_colocated'
TPDIR = './outputs/newTP15'
SAVEDIR = './outputs/N5ESMR_out'
MASKDIR = './reference_files'
AREAFILE = './reference_files/areas_ease_v2atle.cfg'
FIRSTBEAM = 4
LASTBEAM = 74
FIRSTDATE = '1972-12-11'
LASTDATE = '1977-05-16'
TB_ERROR = 3.0

def n5esmr_process(date):
    date = str(date)
    year = date[:4]
    month = date[5:7]
    day = date[8:10]

    subdir = DATADIR + '/ESMR_ERA5_coloc_' + year
    files = glob(subdir + '/Nimbus5-ESMR_' + year + 'm' + month + day + '*')
    TP_filename = os.path.join(TPDIR, 'Nimbus5-ESMR_'+year+'m'+month+day+'.json')
    TP_data = json.load(open(TP_filename))

    if files:
        # loading relevant data from all files to dict and calculating extra parameters
        data = n5esmr.loadParams(files, [FIRSTBEAM,LASTBEAM])
        if not data:
            return

    run4hemisphere(data, TP_data, 'N', [32, 90], date)
    run4hemisphere(data, TP_data, 'S', [-90, -48], date)


# %%
def run4hemisphere(data, TP_data, hemisphere, latlims, date):
    if hemisphere == 'N':
        ice_name = 'arctic_ice'
        ocean_name = 'arctic_ocean'
        area_def = pr.parse_area_file(AREAFILE, 'ease_nh')[0]
        maskfile = os.path.join(MASKDIR, 'LandOceanLakeMask_nh_ease2-250_v2.nc')
    elif hemisphere == 'S':
        ice_name = 'antarctic_ice'
        ocean_name = 'antarctic_ocean'
        area_def = pr.parse_area_file(AREAFILE, 'ease_sh')[0]
        maskfile = os.path.join(MASKDIR, 'LandOceanLakeMask_sh_ease2-250_v2.nc')
    else:
        raise 'Unrecognized hemisphere'

    #specifying current TP_dictionaries
    Tpo_ow = TP_data[ocean_name]
    Tpo_ice = TP_data[ice_name]
    
    #masking atributes to be used based on latlims
    cref = (data['Latitude'] > latlims[0]) & (data['Latitude'] < latlims[1])
    lat = data['Latitude'][cref]
    lon = data['Longitude'][cref]
    Tb = data['Brightness_temperature'][cref]
    vapor = data['tcwv'][cref]
    wind = data['wind'][cref]
    ia = data['incidence_angle'][cref]
    Ts = data['mt2m'][cref]
    Ti = 0.4*data['t2m'][cref]+163.2 #ice correction

    # calculating uncorrected SIC
    SIC_uncorr = (Tb - Tpo_ow['mean_Tb'])/(Tpo_ice['mean_Tb'] - Tpo_ow['mean_Tb'])    
    SIC_uncorr = np.clip(SIC_uncorr, 0.0, 1.0)
    SIC_uncorr[SIC_uncorr < 0.15] = 0.0
    
    # calculating mean atmospheric values
    mvapor  = Tpo_ice['mean_vapor'] *SIC_uncorr + Tpo_ow['mean_vapor'] *(1-SIC_uncorr)
    mwind   = Tpo_ice['mean_wind']  *SIC_uncorr + Tpo_ow['mean_wind']  *(1-SIC_uncorr)
    mliquid = Tpo_ice['mean_liquid']*SIC_uncorr + Tpo_ow['mean_liquid']*(1-SIC_uncorr)
    msst    = Tpo_ice['mean_sst']   *SIC_uncorr + Tpo_ow['mean_sst']   *(1-SIC_uncorr)
    mt2m    = Tpo_ice['mean_t2m']   *SIC_uncorr + Tpo_ow['mean_t2m']   *(1-SIC_uncorr)
    mt2m = 0.4*mt2m + 163.2 # ice correction

    # calculating RTM correction
    Tb_sim_ref = n5esmr.esmr(mvapor, mwind, mliquid, msst, mt2m, SIC_uncorr ,np.abs(ia))
    Tb_sim_act = n5esmr.esmr(vapor,   wind, mliquid,   Ts,   Ti, SIC_uncorr, np.abs(ia))
    dTb_sim = Tb_sim_ref - Tb_sim_act
    Tb_corr = Tb + dTb_sim
    
    #calculating corrected SIC
    SIC_corr =  (Tb_corr - Tpo_ow['mean_Tb_corr'])/(Tpo_ice['mean_Tb_corr'] - Tpo_ow['mean_Tb_corr'])
    SIC_corr_clip = np.clip(SIC_corr, 0.0, 1.0)

    # quantify algorithm error
    delta_Tb_water = (-(1 - SIC_corr_clip)*Tpo_ow['std_Tb_corr'] / (Tpo_ice['mean_Tb_corr'] - Tpo_ow['mean_Tb_corr']))**2
    delta_Tb_ice = (-SIC_corr_clip*Tpo_ice['std_Tb_corr'] / (Tpo_ice['mean_Tb_corr'] - Tpo_ow['mean_Tb_corr']))**2
    algorithm_uncertainty = np.sqrt(delta_Tb_water + delta_Tb_ice)

    # Resampling data
    lon, lat = pr.utils.check_and_wrap(lon, lat)
    swath_def  = pr.geometry.SwathDefinition(lons=lon, lats=lat)
    area_shape = area_def.shape[0]
    fields = ['raw_ice_conc_values', 'algorithm_standard_error', 'Tb_corr', 'Tb', 'Ts']
    data_vars = {}
    for i, atr in enumerate([SIC_corr, algorithm_uncertainty, Tb_corr, Tb, Ts]):
        data_vars[fields[i]] = pr.kd_tree.resample_nearest(swath_def, atr, area_def, radius_of_influence=50000, fill_value=np.nan) 
    SIC_filtered = np.copy(data_vars['raw_ice_conc_values'])
    
    # Calculating smearing and total error
    sic_max = ndimage.maximum_filter(SIC_filtered, size=3)
    sic_min = ndimage.minimum_filter(SIC_filtered, size=3)
    sic_smear = sic_max - sic_min
    sic_smear[np.isnan(SIC_filtered)] = 0.0
    sic_smear[sic_smear > 0.9] = 0.0
    data_vars['smearing_standard_error'] = sic_smear
    data_vars['total_standard_error'] = np.sqrt(data_vars['smearing_standard_error']**2 + data_vars['algorithm_standard_error']**2)
    
    # %% flags
    flag = np.ones([area_shape, area_shape], dtype=np.uint16)*4 #assuming open water filter per default
    
    # load masks
    maskdata = xr.open_dataset(maskfile)
    smask = maskdata['smask_sicci'].to_numpy().astype(int)
    climatology = maskdata['climatology'][int(date[5:7]) - 1].to_numpy()
    climatology[np.isnan(climatology)]=-1
    climatology = climatology.astype(int)
    maskdata.close()
    
    # Temperature flag (16)
    flag[data_vars['Ts']>278.15] = 16

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
    data_vars['total_standard_error'][invalid_surface] = np.nan

    # Invalid point flag (128)
    flag[(flag==4) & np.isnan(SIC_filtered)] = 128
    flag[(flag==4) & (SIC_filtered>=0)] = 0

    # %% prep for data export
    SIC_filtered = np.clip(SIC_filtered,0,1)
    data_vars['status_flag'] = flag
    data_vars['ice_conc'] = SIC_filtered
    del data_vars['Ts']

    makeNetCDF(data_vars, date, hemisphere, area_def, SAVEDIR)

# %% main
if __name__ == "__main__":
    dateRange = np.arange(np.datetime64(FIRSTDATE,'D'),np.datetime64(LASTDATE,'D'))
    for d in tqdm(dateRange):
        n5esmr_process(d)