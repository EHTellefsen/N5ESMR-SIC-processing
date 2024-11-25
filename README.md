# Updated Sea Ice Products for the NIMBUS 5 Electrically Scanning Microwave Radiometer data from 1972–1977

## Motivation
[Kolbe et. al. (2024)](10.5194/essd-16-1247-2024) describes and creates a reprocessed version of sea ice concentration estimates for the NIMBUS 5 Electrically Scanning Microwave Radiometer (N5ESMR) data. 
Since publication, 2 major challenges have been identified, which the code of this repository is made to solve:
- Filtering has been too agressive, removing almost all data past 1975.
- Given N5EMSR only measures the horizontal channel, distinguishing multiyear-ice from lower sea ice concentration is ambiguious.

A new filtering paradigm have therefore been implimented, and while doing so, bugs in the processing have been identified, and the code have been rewritten to simplify future processing. Furthermore, a scheme
for correcting the multi year ice ambiguity have been implimented called "local dynamical tie points" (LDTP). This takes the temporal stability of the local brightness temperature into account, and picks local tie points
from this to predict the sea ice concentration.

## Methods
The methods for the initial method closely follows [Kolbe et. al. (2024)](10.5194/essd-16-1247-2024), except for some changes in filtering. 
The LDTP methods corrects errors appearing in above dataset by identifying local dynamical tie points of sea ice, though identifying stability of the local brightness temperature via 15 day running standard deviation.  

## Data
All data used are found at  [GES DISC](https://disc.gsfc.nasa.gov/datasets/ESMRN5L1_001/summary). To use this code, this data needs to be extracted and transformed to netCDF format according to Kolbe et. al. (2024).
The outputs generated by this code is available at [DTU Data](10.11583/DTU.27835929).

## Dependencies
Packages necessary to run the code are as follows:
- numpy
- scipy
- tqdm
- glob
- cv2
- uuid
- numba
- xarray

## Usage
The code in this repository can be used to create two outputs:
- An updated version of Kolbe et. al. (2024) where filters, flags and error terms have recieved slight corrections, and the code have been rewritten in a more readable format.
- The LDTP corrected version of the above, which corrects the ambiguity of Multi-year-ice versus lower sea ice concentration.

The code is meant to be run in order described below. Be aware, constant "DATADIR" in "n5esmr_tielpoints.py" and "n5esmr_processing_revised.py" need to be changed to reflect the location of the raw orbit data.
The raw Orbit data will need to be grouped in yearly folders and be in the same format as the one used in Kolbe et. al. (2024).

Order of run:
1. n5esmr_tiepoints.py.
2. n5esmr_periodicTiepoints.py
3. n5esmr_processing_revised.py
To also apply LDTP correction:
4. LDTP_periodicStats.py
5. LDTP_run.py

Outputs for periodic stats in LDTP_periodicStats are only for precomputing running statistics, and can be removed once processing is completed.
