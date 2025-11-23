# --Build in--
import warnings

# --Proprietary--

# --Third Party--
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import find_peaks

# %% Constants
MAXREPEATS = 6 #number of times a value can be repeated consecutively
MAXREPEATS_THRESH = 100 #minimum times repeat error can occur before complete removal (tolerance for above)
MINVAL = 90.0 #minimum brightness temperature value
MAXVAL = 310.0 #maximum brightness temperature value
MAX_MEDIAN_DIFFERENCE = 75 # max difference between pixel and median filtered pixel
MAX_REL_ROW_DIFFERENCE = 0.09 #maximum relative difference between two consecutive rows
MAX_REL_ROW_REPEAT_DIFFERENCE = 0.06 #Maximum relative change for zone error filter
MAX_ZONE = 25 # maximum zone width for zone filters

# %% combined filters
def acceptanceFilter(Tb):
    """
    Overall filter for detecting orbits with const column error. This filter returns True if more than threshold in amount of points
    with this error is detected.
    """
    # applying constant column filter for every, and every second pixel
    CC1mask = constColumnFilter(Tb, MAXREPEATS, ReplaceEntireRow=False)
    CC2mask = constColumnFilter(Tb, MAXREPEATS, ReplaceEntireRow=False, separation=2)

    #Checking whether repeats happens often enough
    if np.count_nonzero(CC1mask)>MAXREPEATS_THRESH or np.count_nonzero(CC2mask)>MAXREPEATS_THRESH:
        return True
    else:
        return False

def swathFilter(Tb):
    """
    Applying filters to swath to remove invalid data using select filters and predefined constants.
    """
    # applying value filter to remove invalid value
    Tmask = valueFilter(Tb, vmin = MINVAL, vmax=MAXVAL)
    Tb[Tmask]=np.nan

    # Applying pixel filter and miscalibration filter
    Pmask = pixelFilter(Tb, MAX_MEDIAN_DIFFERENCE)
    Mmask = miscalibrationFilter(Tb, singleThreshold=MAX_REL_ROW_DIFFERENCE, pairThreshold = MAX_REL_ROW_REPEAT_DIFFERENCE, maxwidth=MAX_ZONE)

    # combining masks
    mask = Mmask|Pmask|Tmask

    # using masks for zonefilter
    mask = zoneFilter(mask, maxwidth=MAX_ZONE)

    #applying filters
    Tb[mask]=np.nan



# %% filters
def constColumnFilter(Tb, kernel_length: int, tolerance: float = 0, ReplaceEntireRow: bool = True, separation: int = 1):
    """
    Returns mask where values repeating concecutively for kernel length is set to True. Tolerance allows for slight deviation, and separation denotes
    wether its every concecutive point, every other or so forth. ReplaceEntireRow means whole row is removed and not just repeating pixels.
    """
    # calculating mean difference over columns
    Rowdiff = abs(Tb[:-separation] - Tb[separation:])
    meandiff = uniform_filter(Rowdiff,(kernel_length,1))
    
    # defining mask based on replaceEntireRow attribute
    if ReplaceEntireRow:
        maskKernel = np.any(meandiff<=tolerance, axis=1)
    else:
        maskKernel = meandiff<=tolerance
    maskKernel = maskKernel[kernel_length//2:-kernel_length//2]    

    # applying kernel to all pixels impacted
    diffMask = np.zeros(Rowdiff.shape, dtype=bool)
    for i in range(kernel_length):
        diffMask[i:-kernel_length+i][maskKernel] = True

    # taking into account shift due to difference calcualtion
    mask = np.zeros(Tb.shape, dtype=bool)
    mask[:-separation][diffMask] = True
    mask[separation:][diffMask] = True
    return mask


def valueFilter(Tb, vmin: float, vmax: float):
    """
    Returns mask where all values below or above min or max threshold is true.
    """
    mask = (Tb < vmin)|(Tb > vmax)
    return mask


def pixelFilter(Tb, threshold: float, kernelsize: int = 3):
    """
    Returns mask where values, where difference between value and median filter value is larger than threshold, is set to True.
    """    
    Tb_smoothed = median_filter(Tb, kernelsize)
    Tb_diff = abs(Tb - Tb_smoothed)
    mask = Tb_diff > threshold
    return mask


def miscalibrationFilter(Tb, singleThreshold: float, pairThreshold: float, maxwidth: int):
    """
    Combines rowfilter, miscalibrationFilter_zone, and an ending filter, to avoid calculating change multiple times.
    """
    # calculating relative difference, ignoring errors for nan columns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        relativeChange =  np.nanmedian((Tb[:-1] - Tb[1:])/Tb[:-1], axis=1)

    # finding peaks in change given the threshold in pairs
    peaks = find_peaks(abs(relativeChange), height=pairThreshold)[0]
    signs = np.sign(relativeChange[peaks])

    mask = np.zeros(Tb.shape, dtype=bool)
    for i in range(len(peaks)):
        
        # Difference between 2 peaks smaller than maxwidth
        if i < len(peaks)-1:
            dist = peaks[i+1]-peaks[i]
            if dist<=maxwidth: 
                if signs[i+1]==-signs[i]:
                    mask[peaks[i]:peaks[i+1]+1]=True
                    mask[peaks[i]+1:peaks[i+1]+2]=True
        
        #peak is less than 25 pixels from start
        if peaks[i]<maxwidth:
            mask[:peaks[i]+2]=True
        
        #peak is less than 25 pixels from end
        if len(relativeChange)-peaks[i] < maxwidth: 
            mask[peaks[i]-1:]=True 

         #peak is over single threshold and singular row should be removed
        if abs(relativeChange[peaks[i]])>singleThreshold:
            mask[peaks[i]]=True
            mask[peaks[i]+1]=True 
    return mask



def zoneFilter(inputmask, maxwidth: float):
    """
    Filter for removing data if rows//4 within maxwidth of one another has already been masked, assuming 
    everything in between is likely also error prone.
    """
    height, width = inputmask.shape
    
    #Identifying rows where more than 1/4 have been removed
    removedRows = np.where(np.sum(inputmask, axis=1)>=width//4)[0]
    
    mask = np.zeros(inputmask.shape, dtype=bool)
    if removedRows.size > 0:
        # remove rows in beginning
        if np.min(removedRows) < maxwidth:
            mask[:np.max(removedRows[removedRows < maxwidth])]=True

        # remove rows at end
        if height - np.max(removedRows) < maxwidth:
            mask[np.min(removedRows[height - removedRows < maxwidth]):]=True

        # remove within zonewidth
        for i in range(len(removedRows)-1):
            diff = removedRows[i+1] - removedRows[i]
            if diff < maxwidth:
                mask[removedRows[i]:removedRows[i+1]]=np.nan
    
    return mask

# %% Unused filters
def rowFilter(Tb, threshold: float):
    """
    Returns mask where rows, where absolute relative difference between 2 consecutive rows is higher than threshold, is True.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Rowdiff = np.nanmedian((Tb[:-1] - Tb[1:])/Tb[:-1], axis=1)
    tbfiltered = Rowdiff > threshold
    mask = np.zeros(Tb.shape, dtype=bool)
    mask[:-1][tbfiltered] = True
    mask[1:][tbfiltered] = True
    return mask

def miscalibrationFilter_zone(Tb, threshold: float, maxwidth: int):
    """
    Returns mask where zones experiencing a sudden drop and rise in relative value between 2 consecutive rows are mask out.
    Threshold denotes relative change allowed, and maxwidth is maximum width of zone.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        relativeChange =  np.nanmedian((Tb[:-1] - Tb[1:])/Tb[:-1], axis=1)
    peaks = find_peaks(abs(relativeChange),height=threshold)[0]
    signs = np.sign(relativeChange[peaks])

    mask = np.zeros(Tb.shape, dtype=bool)
    for i in range(len(peaks)-1): 
        if peaks[i+1]-peaks[i]<maxwidth: # Difference between 2 peaks smaller than maxwidth
            if signs[i+1]==-signs[i]:
                mask[peaks[i]:peaks[i+1]]=True
                mask[peaks[i]+1:peaks[i+1]+1]=True

    return mask

# %% Legacy filters
def john_filtering(dataset, brightness_temperature_filtered):
    brt = np.copy(dataset.Brightness_temperature.values)
    brt = np.transpose(brt)
    brt_cleaned = np.copy(brt)
    
    derivative       = brt[0:, 0:-1]-brt[0:, 1:]
    derivative_m     = np.nanmean(np.absolute(derivative), axis=0)
    derivative_3     = np.absolute(derivative[0:, 0:-1]) + np.absolute(derivative[0:, 1:])
    derivative_horz  = brt[0:-1, 0:] - brt[1:, 0:]
    derivative_horz_3 = np.absolute(derivative[0:-1, 0:]) + np.absolute(derivative[1:, 0:])
    derivative_vert_zeros = np.absolute(derivative[0:,0:-4]) + np.absolute(derivative[0:,1:-3]) + np.absolute(derivative[0:,2:-2]) + np.absolute(derivative[0:,3:-1]) + np.absolute(derivative[0:,4:])
    derivative_horz_zeros = np.absolute(derivative[0:-2, 0:]) + np.absolute(derivative[1:-1, 0:]) + np.absolute(derivative[2:,0:])
    
    mask_vert = derivative_vert_zeros == 0
    D3_bound = 150
    
    mask2 = derivative_m > 50 # reates a boolean of size derivative_m where 1 indicatets values over 50
    mask2 = np.tile(mask2, (brt.shape[0], 1)) # creates a matrix of size brightness out of mask 2 vector
    mask1  = derivative_3 > D3_bound
    mask1 = np.amax(mask1, axis=0) == 1 # max returns a vector of rows with a value of one
    mask1 = np.tile(mask1, (brt.shape[0], 1)) # creates a matrix from the vectors of mask1
    mask2_f = np.zeros((brt.shape[0], brt.shape[1]), dtype=bool) # creates matrix of size brt
    mask2_f[0:,0:-1] = mask2_f[0:,0:-1] + mask2 # over writes values to 1 if true in the mask for 1 to second to last
    mask2_f[0:,1:] = mask2_f[0:,1:] + mask2 #over writes values to 1 if true in the mask for 2 to last. removing both rows if their diffrence is greater then 50
    mask1_f = np.zeros((brt.shape[0], brt.shape[1]), dtype=bool) #creates matrix of size brt
    mask1_f[0:,0:-2] = mask1_f[0:,0:-2] + mask1 #becouse mask1 is 3 indeceas shorter then brt for each error it removes the 3 rows used to calculate derivative_3
    mask1_f[0:,1:-1] = mask1_f[0:,1:-1] + mask1
    mask1_f[0:,2:] = mask1_f[0:,2:] + mask1
    mask_high = brt > 300 #  makes boolean sized brt with true greater then 300k 
    mask_low = brt <= 90 #  makes boolean sized brt with <=90 k 
    mask_vert_f = np.zeros((brt.shape[0], brt.shape[1]), dtype=bool)
    mask_vert_f[0:,0:-5] = mask_vert_f[0:, 0:-5] + mask_vert
    mask_vert_f[0:,1:-4] = mask_vert_f[0:, 1:-4] + mask_vert
    mask_vert_f[0:,2:-3] = mask_vert_f[0:, 2:-3] + mask_vert
    mask_vert_f[0:,3:-2] = mask_vert_f[0:, 3:-2] + mask_vert
    mask_vert_f[0:,4:-1] = mask_vert_f[0:, 4:-1] + mask_vert
    mask_vert_f[0:,5:] = mask_vert_f[0:, 5:] + mask_vert
    mask_horz_f = np.zeros((brt.shape[0], brt.shape[1]), dtype=bool)
    
    
    brightness_temperature_filtered[np.transpose(mask_high)]     = np.nan # sets values from each mask to NaN
    brightness_temperature_filtered[np.transpose(mask_low)]      = np.nan
    brightness_temperature_filtered[np.transpose(mask2_f)]     = np.nan
    brightness_temperature_filtered[np.transpose(mask1_f)]     = np.nan
    brightness_temperature_filtered[np.transpose(mask_vert_f)]   = np.nan



def analog_filter(dataset, brightness_temperature_filtered):
    GRADIENT_THRESHOLD = 10

    mask = np.zeros(brightness_temperature_filtered.shape[0], dtype=bool)
    for i in range(0, 15):
        temp = np.where( np.absolute( np.gradient(dataset['Analog_' + str(i)])) > GRADIENT_THRESHOLD)   
        temp = np.unique([temp[0], temp[0]-1, temp[0]-2, temp[0]-3, temp[0]-4, temp[0]-5, temp[0]+1, temp[0]+2, temp[0]+3, temp[0]+4, temp[0]+5])
        temp = temp % brightness_temperature_filtered.shape[0] #wk:python works with negative indices, like -1, but not with exceeding ones like len(BT) + 1 -> which gives an error - the % len(BT) converts the indexes to the correct range of len(BT)
        mask[temp] = True

    mask = np.transpose(np.tile(mask, (78, 1)))
    brightness_temperature_filtered[mask] = np.nan 



def beam_position_filter(beam_position):
    result = False
    mask_low = np.where(beam_position < 2)
    mask_high = np.where(beam_position > 1022)
    if ((np.count_nonzero(mask_low) > (beam_position.shape[0] / 4)) or (np.count_nonzero(mask_high) > (beam_position.shape[0] / 4))):
        result = True
    return result