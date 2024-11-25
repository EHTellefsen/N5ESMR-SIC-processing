"""
Running mean correction for daily tie point product created using tiepoints.py.

Author: Emil Haaber Tellefsen
Co-Author: Wiebke Marghitta Kolbe, Rasmus Tage Tonboe

Date: 04/11/2024
"""

# --Build in--
import os
import json

# --Third Party--
import numpy as np

# %% constants
TPDIR = './outputs/newTP'
SAVEDIR = './outputs/newTP15'
FIRSTDATE = '1972-12-11'
LASTDATE = '1977-05-16'
PERIOD_LENGTH = 15

def periodicTiepoints(dateRange):
    N = len(dateRange)

    # %% loading data
    # load sample file to learn keys
    file =os.path.join(TPDIR,os.listdir(TPDIR)[0])
    initfile = json.load(open(file))
    majorKeys = initfile.keys()
    minorKeys = initfile['arctic_ice'].keys()

    # create data storage
    fullStore = {}
    for majorKey in majorKeys:
        if majorKey == 'filename':
            fullStore[majorKey]=np.empty(N,dtype=object)
        else:
            fullStore[majorKey] = {}
            for minorKey in minorKeys:
                fullStore[majorKey][minorKey] = np.empty(N)

    # running over all dates
    for i, date in enumerate(dateRange):
        date = str(date)
        year = date[:4]
        month = date[5:7]
        day = date[8:10]
        
        filename = 'Nimbus5-ESMR_%sm%s%s.json' % (year, month, day)

        #checking if file is available for given date
        file = os.path.join(TPDIR,filename)
        if os.path.isfile(file):
            data = json.load(open(os.path.join(TPDIR,filename)))
            
            # adding data to major dictionary
            for majorKey in majorKeys:
                if majorKey == 'filename':
                    fullStore[majorKey][i]= filename
                else:
                    for minorKey in minorKeys:
                        fullStore[majorKey][minorKey][i] = data[majorKey][minorKey]

        #replacing with NaN in case no data is available        
        else:
            for majorKey in majorKeys:
                if majorKey == 'filename':
                    fullStore[majorKey][i]= filename
                else:
                    for minorKey in minorKeys:
                        fullStore[majorKey][minorKey][i] = np.nan       


    # %% Calculating means and saving
    halfP = PERIOD_LENGTH//2

    for i, date in enumerate(dateRange):
        date = str(date)
        year = date[:4]
        month = date[5:7]
        day = date[8:10]
        filename = 'Nimbus5-ESMR_%sm%s%s.json' % (year, month, day)

        outDict = {}
        for majorKey in majorKeys:
            if majorKey == 'filename':
                outDict[majorKey]=filename
            else:
                outDict[majorKey] = {}
                for minorKey in minorKeys:
                    outDict[majorKey][minorKey] = np.nanmean(fullStore[majorKey][minorKey][max(0,i-halfP):min(i+halfP,N-1)])

            jf = os.path.join(SAVEDIR,filename)
            with open(jf, 'w') as fp:
                json.dump(outDict, fp, indent=4)
            fp.close()    



# %% main
if __name__ == "__main__":
    dateRange = np.arange(np.datetime64(FIRSTDATE,'D'),np.datetime64(LASTDATE,'D'))
    periodicTiepoints(dateRange)
