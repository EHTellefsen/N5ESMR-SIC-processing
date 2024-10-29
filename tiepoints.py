# --Build in--
import os
import json
import sys

# --Proprietary--
sys.path.append('./functions')
import n5esmr

# --Third Party--
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm


# %% constants
DATADIR = 'D:/N5ESMR/ESMR_ERA5_colocated'
SAVEDIR = './outputs/newTP'
FIRSTBEAM = 4
LASTBEAM = 74
FIRSTDATE = '1972-12-11'
LASTDATE = '1972-12-13'#'1977-05-16'


# %% Tie point main function
def tiepoint(date):
    # locating relevant files
    date = str(date)
    year = date[:4]
    month = date[5:7]
    day = date[8:10]
    filename_out = 'Nimbus5-ESMR_'+year+'m'+month+day

    subdir = DATADIR + '/ESMR_ERA5_coloc_' + year
    files = glob(subdir + '/Nimbus5-ESMR_' + year + 'm' + month + day + '*')
    if files:
        # loading relevant data from all files, filter, and add to dict and calculating extra parameters
        data = n5esmr.loadParams(files, [FIRSTBEAM,LASTBEAM])
        if not data:
            return

        # %% calculation of tiepoints
        #smearing SIC via uniform filter
        kernel55 = np.ones((5,5),np.float32)/25
        sic55 = cv2.filter2D(data['siconc'],-1,kernel55)

        # Finding tiepoint references for ocean
        no_ice = (data['siconc'] == 0)
        away_from_ice_edge = (sic55 < 0.01) 
        ocean_tb_low = (data['Brightness_temperature'] > 50.0)
        ocean_tb_high = (data['Brightness_temperature'] < 180.0)
        cold_water = (data['sst'] < 278.0)
        ocean_tp_ref = no_ice & away_from_ice_edge & cold_water & ocean_tb_low & ocean_tb_high

        # Finding tiepoint references for ice
        ice=(data['siconc'] > 0.8)
        within_ice=(sic55 > 0.8)
        ice_tb_low=(data['Brightness_temperature'] > 100.0)
        ice_tb_high=(data['Brightness_temperature'] < 274.0)
        ice_tp_ref =ice & within_ice & ice_tb_low & ice_tb_high

        # apply RTM for all surface types
        TP_NH_ice = applyRTM(data, ice_tp_ref, 'N', 'ice')
        TP_SH_ice = applyRTM(data, ice_tp_ref, 'S', 'ice')
        TP_NH_ocean = applyRTM(data, ocean_tp_ref, 'N', 'ocean')
        TP_SH_ocean = applyRTM(data, ocean_tp_ref, 'S', 'ocean')

        # merging area dictionaries
        tpOut = {"filename": filename_out,
                'arctic_ice':TP_NH_ice,
                'antarctic_ice':TP_SH_ice,
                'arctic_ocean':TP_NH_ocean,
                'antarctic_ocean':TP_SH_ocean
                }
        
        # Saving json file
        jf = os.path.join(SAVEDIR,filename_out+'.json')
        with open(jf, 'w') as fp:
            json.dump(tpOut, fp, indent=4)
        fp.close()


# %% applying RTM
def applyRTM(data, ref, hemisphere, surface):
    # Applies RTM correction and find tiepoint values given area, hemisphere and 
    # surface type (ice or ocean)
    
    # variables in output file
    variables = ["median_Tb", "mean_Tb", "std_Tb", "mean_Tb_corr",
                 "std_Tb_corr", "mean_correction", "max_correction", "min_correction",
                 "mean_t2m", "mean_sst", "mean_vapor", "mean_liquid", "mean_wind"
                ]

    # picking relevant area depending on hemisphere
    if hemisphere == 'N':
        cref = (data['Latitude'] >= 0) & ref
    elif hemisphere == 'S':
        cref = (data['Latitude'] <= 0) & ref
    else:
        raise 'Unrecognized hemisphere'

    # making output dictionary and calculating number of points
    TP = {}
    N = np.count_nonzero(cref)
    TP["number_of_points"] = N

    # Setting all atributes to NaN per default - output when N<10
    for v in variables:
        TP[v]=np.nan
    
    if N > 10:
        #defining relevant variables
        cTb = data['Brightness_temperature'][cref]
        cvapor = data['tcwv'][cref]
        cwind = data['wind'][cref]
        cclw = data['clw'][cref]
        cia = data['incidence_angle'][cref]

        #saving variable statistics
        TP["median_Tb"] = np.median(cTb)   
        TP["mean_Tb"] = np.mean(cTb)
        TP["std_Tb"] = np.std(cTb)
        TP["mean_sst"] = np.mean(data['mt2m'][cref])
        TP["mean_vapor"] = np.mean(cvapor)
        TP["mean_liquid"] = np.mean(cclw)
        TP["mean_wind"] = np.mean(cwind)     
    
        if surface=='ice':
            #defining ice and surface temperature
            cTi = data['t2m'][cref]
            cTs = data['t2m'][cref]
            TP['mean_t2m'] = np.mean(cTi)
            #ice correction
            cTi = 0.4*cTi + 0.6*272.0
            cSIC = 1.0
        elif surface=='ocean':
            #defining ice and surface temperature
            cTi =data['mt2m'][cref]
            cTs = data['mt2m'][cref]
            TP['mean_t2m'] = np.mean(cTi)
            cSIC = 0.0 
        else:
            raise 'Unrecognized surface'
        
        #simulating Tb via RTM
        cTb_sim_act = n5esmr.esmr(cvapor, cwind, TP["mean_liquid"], cTs, cTi, cSIC, np.abs(cia))
        cTb_sim_ref = n5esmr.esmr(TP["mean_vapor"], TP["mean_wind"], TP["mean_liquid"], TP["mean_t2m"], np.mean(cTi), cSIC, np.abs(cia))

        #calculating corrected Tb
        dcTb_sim = cTb_sim_ref - cTb_sim_act
        Tb_corr = cTb + dcTb_sim 

        #Recording corrections
        TP["mean_Tb_corr"] = np.mean(Tb_corr)
        TP["std_Tb_corr"] = np.std(Tb_corr)
        TP["mean_correction"] = np.mean(dcTb_sim)
        TP["min_correction"] = np.min(dcTb_sim)
        TP["max_correction"] = np.max(dcTb_sim)
    return TP


# %% main
if __name__ == "__main__":
    dateRange = np.arange(np.datetime64(FIRSTDATE,'D'),np.datetime64(LASTDATE,'D'))
    for d in tqdm(dateRange):
        tiepoint(d)