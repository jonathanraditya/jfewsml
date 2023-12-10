# import modules
from sqlite3 import connect as con
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime as dtime
from datetime import timedelta as tdelta
import pandas as pd
import numpy as np
import random
from skimage import color
import h5py
import time
import tensorflow as tf
from sklearn.metrics import r2_score
import hydroeval
import keras
from PIL import Image
import copy

class MetadataInfo:
    # Sadewa base URL
    s_url = 'https://sadewa.sains.lapan.go.id/'
    # Sadewa Source Information
    s_sinfo = {
        'IR1':{
            'url':s_url+'HIMAWARI/himawari_merc/IR1/{}/{}/{}/',
            'fname':'H89_IR1_{}{}{}{}00.png',
            'yearStart':'2020'
        },
        'IR3':{
            'url':s_url+'HIMAWARI/himawari_merc/IR3/{}/{}/{}/',
            'fname':'H89_IR3_{}{}{}{}00.png',
            'yearStart':'2020'
        },
        'VIS':{
            'url':s_url+'HIMAWARI/himawari_merc/VIS/{}/{}/{}/',
            'fname':'H89_VIS_{}{}{}{}00.png',
            'yearStart':'2020'
        },
        'B04':{
            'url':s_url+'HIMAWARI/himawari_merc/B04/{}/{}/{}/',
            'fname':'H89_B04_{}{}{}{}00.png',
            'yearStart':'2020'
        },
        'CCLD':{
            'url':s_url+'HIMAWARI/komposit/{}/{}/{}/',
            'fname':'H89_CCLD_{}{}{}{}00.png',
            'yearStart':'2020'
        },
        'rain':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'rain_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'cloud':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'cloud_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'psf':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'psf_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'qvapor':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'qvapor_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'sst':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'sst_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'wind':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'wind_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'winu':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'winu_{}{}{}_{}.png',
            'yearStart':'2019'
        },
        'wn10':{
            'url':s_url+'wrf/{}/{}/{}/',
            'fname':'wn10_{}{}{}_{}.png',
            'yearStart':'2019'
        },
    }
    def __init__(self):
        pass


mi = MetadataInfo()
        

def showSampleSadewaData(hdtype='cloud',idx=0):
    '''
    showing sadewa sampel data 
    current : cloud-0
    '''
    path = os.path.join(os.path.dirname(os.getcwd()), 
                        'mining_sadewa', 'sadewa',
                        hdtype, getHimawariFilename()[hdtype][idx])
    img = mpimg.imread(path)
    plt.imshow(img, cmap='rainbow')
    plt.colorbar()
    
def printSadewaFetchMissingData(fn='sadewaerr.txt'):
    '''
    printing total count of missing data for every sadewa-himawari dataset
    '''
    p = os.path.join(os.path.dirname(os.getcwd()), 'mining_sadewa', fn)
    with open(p, 'r') as f:
        c = f.read()
    
    err = c.split('.png')
    err = [e + '.png' for e in err]
    
    errlist = {k:[] for k in list(mi.s_sinfo.keys())}

    # loop to append error URL for each data type
    for e in err:
        # loop for each key to check for True value
        for k in errlist.keys():
            # check for match data type
            if k in e:
                errlist[k].append(e)

    # print count of missing data for each data 
    for k in errlist.keys():
        print(e, len(errlist[k]))
        
def idealDataCount():
    '''
    returning (2019idc, 2020idc) ideal data count for Himawari dataset
    '''
    # calculating ideal data count for each entry date
    early2019=[2019,1,1]
    early2020=[2020,1,1]
    minedDate=[2021,3,14]
    fdE19= dtime(*(early2019))
    fdE20= dtime(*(early2020))
    fdMD= dtime(*(minedDate))

    dateRange2019=(fdMD-fdE19).days
    dateRange2020=(fdMD-fdE20).days

    # ideal data count for each entry date
    dataCount2019=dateRange2019*24
    dataCount2020=dateRange2020*24
    
    return dataCount2019, dataCount2020

def manggaraiFullData(db='dsda.db'):
    # read and fetch database data to pandas dataframe
    # dsdaPath = os.path.join(os.path.dirname(os.getcwd()), 'mining_dsda', db)
    
    df = pd.read_sql_query('SELECT * FROM manggarai', con(db))
    df = df.convert_dtypes()
    df.set_index('currentdate')
    df['currentdate'] = df['currentdate'].astype('datetime64[ns]')
    return df

def manggaraiDataList(maxData=True, hourOffset=0, wlstation='manggarai', db='dsda.db', start_date='2019-02-01 00:00'):
    '''
    Returning a tuple of list (date, data) of manggarai TMA data with 10-minutes-interval from DSDA dataset in year 2020
    '''
    df = manggaraiFullData(db=db)
    df = df.loc[df['currentdate'] >= start_date]

    df['data'] = df['data'].astype('int')
    gpd = df.groupby([df['currentdate'].dt.date, df['currentdate'].dt.hour])['data'].max()
    
    dt, v = [], []
    for x in gpd.items():
        dt.append(dtime(x[0][0].year, x[0][0].month, x[0][0].day, x[0][1]))
        v.append(x[1])
    return dt, v

def getHimawariFilename():
    '''
    Return dictionary of available himawari data based on filename inside
    folder as a key
    '''
    RPATH = os.getcwd()
    hpath = os.path.join(os.path.dirname(RPATH), 'mining_sadewa', 'sadewa')
    
    return {d:os.listdir(os.path.join(hpath, d)) for d in os.listdir(hpath)}

def extractHimawariDatetime(usedData, obs_list=['CCLD','B04','IR1','IR3','VIS'], pred_list=['cloud','psf','qvapor','rain','sst','wind','winu','wn10'], intersection=True):
    '''
    Extract every filename in sadewa-himawari data to datetime object for easier handling

    if intersection=True, it returns a list with an intersection datetime object availability with other data
    
    Returns :
    extractedDate -- dictionary containing list of datetime object for each filename inside dictionary keys for every data
    '''
    himawari=getHimawariFilename()

    # extract date for each himawari data type to datetime.datetime object
    
    results = {}
    for var in usedData:
        if var in obs_list:
            results[obs]=[
                dtime.strptime(x.replace('H89_{}_'.format(obs),'')
                               .replace('.png',''), '%Y%m%d%H%M')
                for x in himawari[obs]]
                
        elif var in pred_list:
            results[var]=[
                dtime.strptime(x.replace('{}_'.format(pred),'')
                               .replace('.png','')
                               .replace('_','')+'00', '%Y%m%d%H%M') 
                for x in himawari[pred]]
        else:
            print(f'Check `usedData` args. {var} is not found in data list.')
              
    return results if not intersection else list(set(results[usedData[0]]).intersection(*list(results.values())))

def getAvailableSlicedData(maxData=True, hourOffset=0, dataScope='combination', wlstation='manggarai'):
    '''
    check through all available dataset, including manggarai TMA, sadewa-himawari IR1, IR3, VIS, B04, and CCLD
    and return a tuple containing datetime object and manggarai hourly TMA data that are synced through all available dataset
    
    This function doesn't return sadewa-himawari data, because using the datetime format and the sadewa-himawari data types,
    the full name of the file required can be constructed.
    
    return : (slicedDate, slicedData) # both are lists inside a tuple
    '''
    if dataScope == 'combination':
        usedData=['CCLD','B04','IR1','IR3','VIS','rain','cloud','psf','qvapor','sst']
    elif dataScope == 'prediction':
        usedData=('cloud','psf','qvapor','rain','sst','wind','winu','wn10')
    
    extractedDate = extractHimawariDatetime(usedData=usedData)
        
    # getting date-data slice from himawari and manggarai TMA data

    # using function to get manggarai available date-data
    dt, v = manggaraiDataList(maxData, hourOffset, wlstation=wlstation)

    df = pd.DataFrame({'dt':dt,'v':v})
    df = df.loc[df.dt.isin(extractedDate)]

    return pd.to_datetime(df.dt).tolist(), df.v.tolist()

def statisticsRaw(maxData=True):
    '''
    Return pandas dataframe of statistics in all available data
    
    column 0 : date
    column 1 : tma
    column 2 - 152 : obs/pred * dataset * med/mean/stdev/min/max
    '''
    adte, adta = getAvailableSlicedData(maxData)

    himawariData = {'o100' : {'fname' : 'observation100',
                              'dataset' : ['CCLD','B04','IR1','IR3','VIS']},
                    'o196' : {'fname' : 'observation196',
                              'dataset' : ['CCLD','B04','IR1','IR3','VIS']},
                    'o400' : {'fname' : 'observation400',
                              'dataset' : ['CCLD','B04','IR1','IR3','VIS']},
                    'p100' : {'fname' : 'prediction100',
                              'dataset' : ['rain','cloud','psf','qvapor','sst']},
                    'p196' : {'fname' : 'prediction196',
                              'dataset' : ['rain','cloud','psf','qvapor','sst']},
                    'p400' : {'fname' : 'prediction400',
                              'dataset' : ['rain','cloud','psf','qvapor','sst']}}

    df = {'date':adte,
          'tma':adta}
    dtDF = pd.DataFrame(df)
    dtDF['tma'] = dtDF['tma'].astype('int64')

    for himawari in himawariData:

        # start statistics
        tick = time.time()

        # initialize new list for each column
        statistics=[[[],[],[],[],[]],
                   [[],[],[],[],[]],
                   [[],[],[],[],[]],
                   [[],[],[],[],[]],
                   [[],[],[],[],[]]]
        statisticsHeader=['med','mean','stdv','min','max']

        fname = himawariData[himawari]['fname']
        dataset = himawariData[himawari]['dataset']

        # print current dataset
        print(dataset)

        # open file
        with h5py.File('{}.hdf5'.format(fname), 'r') as f:
            fetchData = f['datas'][()]


        # loop for each data row
        for i in range(len(fetchData)):

            # loop for each dataset
            for j in range(len(fetchData[i])):
                # fetch image data
                imageData = fetchData[i][j]

                # convert rgba to rgb
                rgb = color.rgba2rgb(imageData)

                statistics[0][j].append(np.median(rgb)) 
                statistics[1][j].append(np.mean(rgb))
                statistics[2][j].append(np.std(rgb))
                statistics[3][j].append(np.min(rgb))
                statistics[4][j].append(np.max(rgb))

        # end statistics
        tock = time.time()
        print('Elapsed time : {}'.format(tock-tick))

        print('Inserting to dataframe')

        # after fetching statistics value for each dataset, insert to pandas dataframe
        # loop over statistics data array
        for i in range(len(statistics)):
            statHeader = statisticsHeader[i]
            # loop over dataset inside statistics data array
            for j in range(len(dataset)):
                datasetHeader = dataset[j]

                # constructing header name
                header = '{}_{}_{}'.format(fname, datasetHeader, statHeader)

                # append to existing dataframe
                dtDF[header] = statistics[i][j]
                
    return dtDF

# FUNCTIONS #

def cropImageData(imgCropX, imgCropY, adte, usedDatas, imgPath, predData=False):
    '''
    Crop image data based on defined crop bound in horizontal (x) and vertical (y) direction,
    and append the cropped data to nd numpy array with format : (m datas, datatypes, imgdim1, imgdim2, number of channels)
    
    Parameters :
    imgCropX -- list of start and end bound of horizontal slice index image numpy array
    imgCropY -- list of start and end bound of horizontal slice index image numpy array
    adte -- list of available date in datetime object
    usedDatas -- list of want-to-crop data
    imgPath -- complete image path with string format placeholder relative from current working directory
    datef -- main date formatted to inserted into placeholder in imgPath
    dateh -- optional date format for prediction data
    
    Returns :
    croppedData -- numpy array of cropped data with format : (m datas, datatypes, imgdim1, imgdim2, number of channels)
    '''
    # loop conditional
    firstColumn=True
    i=0
    for date in adte:
        # loop conditional
        firstRow=True
        for data in usedDatas:
            if predData:
                datef = date.strftime('%Y%m%d')
                dateh = date.strftime('%H')
            else:
                datef = date.strftime('%Y%m%d%H%M')
                dateh = None

            imgPathF=imgPath.format(data, data, datef, dateh)
            # fetching image data
            #print(imgPath)
            image=mpimg.imread(imgPathF)
            # cropping image to defined dimension(s)
            image=image[imgCropX[0]:imgCropX[1], imgCropY[0]:imgCropY[1]]
            
            image=image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            
            # check for first loop 
            if firstRow:
                sameDate=np.copy(image)
                firstRow=False
            else:
                sameDate=np.vstack((sameDate, image))
        
        # reshaping numpy array
        sameDate=sameDate.reshape(1, sameDate.shape[0], sameDate.shape[1], sameDate.shape[2], sameDate.shape[3])
        
        # check for first loop
        if firstColumn:
            croppedData=np.copy(sameDate)
            firstColumn=False
        else:
            croppedData=np.vstack((croppedData, sameDate))
        if i%100 == 0:
            print(croppedData.shape)
        i+=1
            
    return croppedData 
    

def cropImagePredictionData(dim, usedDatas=['rain','cloud','psf','qvapor','sst'], hourOffset=0, wlstation='manggarai'):
    '''
    Returning numpy array with dimension of (m training data, nodes), that nodes = (rain, cloud, psf, qvapor, sst) cropped data
    based on defined dimensions : 100 (10x10), 196 (14x14), 400 (20x20)
    '''
    if dim == 72:
        imgCropX=[324, 336] # 12x6
        imgCropY=[234, 240] # 12x6
    elif dim == 100:
        imgCropX=[323,333] # 10x10
        imgCropY=[233,243] # 10x10
    elif dim == 196:
        imgCropX=[320,334] # 14x14
        imgCropY=[230,244] # 14x14
    elif dim == 240:
        imgCropX=[318, 338] # 20x12
        imgCropY=[231, 243] # 20x12  
    #elif dim == 400:
        #imgCropX=[317,337] # 20x20
        #imgCropY=[227,247] # 20x20
    elif dim == 400:
        imgCropX=[318, 338] # 20x20v2 shifted down 1 cell
        imgCropY=[227, 247] # 20x20v2 shifted down 1 cell
    # Katulampa crop extent
    elif dim == 16: # Katulampa 4x4 input cell
        imgCropX=[332, 336]
        imgCropY=[236, 240]
    elif dim == 784: # Katulampa 28x28 input cell
        imgCropX=[320, 348]
        imgCropY=[224, 252]
    
    adte, adta = getAvailableSlicedData(dataScope='prediction', hourOffset=hourOffset, wlstation=wlstation)
    imgPath = os.path.join(os.path.dirname(os.getcwd()), 'mining_sadewa', 'sadewa', '{}', '{}_{}_{}.png')

    return cropImageData(imgCropX, imgCropY, adte, usedDatas, imgPath, predData=True)

def cropImageObservationData(dim, usedDatas=['IR1','IR3','B04','VIS','CCLD']):
    '''
    Returning 3 dimensions numpy array with (m training data, nodes), that nodes = (IR1, IR3, B04, VIS, CCLD) cropped data
    based on defined dimensions : 100 (10x10), 196 (14x14), 400 (20x20)
    '''
    if dim == 45:
        imgCropX=[932,941]
        imgCropY=[517,522]
    elif dim == 100:
        imgCropX=[910,920]
        imgCropY=[405,415]
    elif dim == 196:
        imgCropX=[908,922]
        imgCropY=[403,417]
    elif dim == 198: #v2
        imgCropX=[928,942]
        imgCropY=[512,526]
    elif dim == 400:
        imgCropX=[905,925]
        imgCropY=[400,420]
       
    adte, adta = getAvailableSlicedData()
    imgPath = os.path.join(os.path.dirname(os.getcwd()), 'mining_sadewa', 'sadewa', '{}', 'H89_{}_{}.png')

    return cropImageData(imgCropX, imgCropY, adte, usedDatas, imgPath)

def performIndividualCropPredictionData(dim):
    '''
    Perform individual database creation of prediction data
    '''
    # initializing individual variables
    usedDatas = [['cloud'],['psf'],['qvapor'],['rain'],['sst'],['wind'],['winu'],['wn10']]
    for usedData in usedDatas:
        
        crop = cropImagePredictionData(dim, usedDatas=usedData)
        p = os.path.join(os.getcwd(), 'dataset', ''.join(usedData) + dim + '.hdf5' )
        with h5py.File(p, 'w') as f:
            f.create_dataset('datas', data=crop)
    

def dataLabel(entryBound, endBound, dim=100):
    '''
    Returning ndarray of input and label by specifying entry and end bound of available data
    '''
    dateBound=adte[entryBound:endBound]
    labels=np.array(adta[entryBound:endBound]).astype('int16')
    labels=labels.reshape(1, labels.shape[0])

    if dim == 100:
        imgCropX=[910,920]
        imgCropY=[405,415]
    elif dim == 196:
        imgCropX=[908,922]
        imgCropY=[403,417]
    elif dim == 400:
        imgCropX=[905,925]
        imgCropY=[400,420]

    # only used IR1 (top cloud temp), IR3 (water vapor), and CCLD (cloud growth)
    usedDatas=['IR1','IR3','CCLD']

    totalNodes=(imgCropX[1]-imgCropX[0])*(imgCropY[1]-imgCropY[0])*len(usedDatas)
    totalTrainingData=endBound-entryBound

    first=True
    for date in dateBound:
        reshaped=np.array([])
        for data in usedDatas:
            datef=date.strftime('%Y%m%d%H%M')
            imgPath = os.path.join(os.path.dirname(os.getcwd()), 'mining_sadewa', 'sadewa', data, f'H89_{data}_datef.png')
            image=color.rgb2gray(color.rgba2rgb(mpimg.imread(imgPath)))
            image=image[imgCropX[0]:imgCropX[1], imgCropY[0]:imgCropY[1]]
            reshapedP=image.reshape(image.shape[0]*image.shape[1])
            reshaped=np.append(reshaped, reshapedP)
            
        transposed=reshaped.reshape(reshaped.shape[0],1)

        if first:
            trainingData=np.copy(transposed)
            first=False
        else:
            trainingData=np.hstack((trainingData,transposed))
            
    return trainingData, labels

def storeDataLabelHDF5(filename, datas, labels):
    '''
    Store data to HDF5 format to prevent prefetching from scratch
    '''
    with h5py.File(filename, 'w') as f:
        f.create_dataset('datas', data=datas)
        f.create_dataset('labels', data=labels)
        
def readDataLabelHDF5(filename):
    '''
    Read stored data -- and -- label data in HDF5 format, back to numpy array
    '''
    with h5py.File(filename, 'r') as f:
        data=f['datas'][()]
        labels=f['labels'][()]
    return data, labels

def coordinatesTable(sRight, sLeft, sBottom, sTop, resW, resH):
    '''
    Returning coordinates table according to defined parameters
    
    Parameters :
    sRight -- most right bound of the coordinates in decimal degrees
    sLeft -- most left bound of the coordinates in decimal degrees
    sBottom -- most bottom bound of the coordinates in decimal degrees
    sTop -- most top bound of the coordinates in decimal degrees
    resW -- image resolution in vertical direction (in pixels)
    resH -- image resolution in horizontal direction (in pixels)
    
    Returns :
    coordX -- 1d numpy array of entry coordinates for each pixel in result image in X coordinates
    coordY -- 1d numpy array of entry coordinates for each pixel in result image in Y coordinates
    '''
    # maximum value for latitude/longitude coordinates
    maxWidth=180
    maxHeight=90

    # calculating width and height in decimal degrees
    if sLeft >= 0 & sRight < 0:
        sWidth=(maxWidth-sLeft)+(maxWidth+sRight)
    elif sLeft >= 0 & sRight >=0:
        sWidth = sRight - sLeft
    else:
        raise Exception("Condition haven't been defined. Please define first")
    sHeight=sTop-sBottom
    print(sWidth, sHeight)
   
    # initialize pixel index in coordinates (x,y)
    coordX = np.zeros(resW)
    coordY = np.zeros(resH)
    
    # calculating X and Y coordinates for each picture pixel
    for y in range(len(coordY)):
        yCoord=sTop-y/len(coordY)*sHeight
        coordY[y]=yCoord
    for x in range(len(coordX)):
        xCoord=sLeft+x/len(coordX)*sWidth
        # check for timezone pass 
        if not(xCoord < 180):
            xCoord=-180+(xCoord-180)
        coordX[x]=xCoord
        
    return coordX, coordY

def coordinatesPredictionTable():
    '''
    Creating prediction data image coordinates for each pixel at entry point
    
    The (0,0) entry point is located on the top left corner of the image
    '''
    # sadewa prediction data bound
    sRight = 145
    sLeft = 95
    sBottom = -10
    sTop = 10
    
    # sadewa image data resolution
    resW = 1000
    resH = 400
        
    coordX, coordY = coordinatesTable(sRight, sLeft, sBottom, sTop, resW, resH)
        
    return coordX, coordY

def coordinatesObservationTable():
    '''
    Creating observation data image coordinates for each pixel at entry point
    
    The (0,0) entry point is located on the top left corner of the image
    
    Returns :
    coordX -- 1d numpy array of entry coordinates for each pixel in result image in X coordinates
    coordY -- 1d numpy array of entry coordinates for each pixel in result image in Y coordinates
    '''
    # sadewa observation data bound
    sRight=-150
    sLeft=70
    sBottom=-60
    sTop=60

    # sadewa image data resolution
    resW=1565
    resH=1686
        
    coordX, coordY = coordinatesTable(sRight, sLeft, sBottom, sTop, resW, resH)
        
    return coordX, coordY

def crop(coordX, coordY, right=107.2, left=106.5, bottom=-6.7, top=-6.2):
    '''
    Crop Sadewa IR1, IR2, VIS, CCLD, B04 data
    with right,left,bottom, and top coordinates bound (in deg)
    
    Paramters :
    *default value will crop the image to 10x10 pixels
    coordX -- coordinates table in horizontal direction for each pixel
    coordY -- coordinates table in vertical direction for each pixel
    
    Returns :
    resx -- A numpy array containing index (or pixel) in horizontal direction of cropped image
    resy -- A numpy array containing index (or pixel) in vertical direction of cropped image
    '''
    # creating Boolean list to crop image
    yEntryTruthValues = coordY < top
    yEndTruthValues = coordY > bottom
    xEntryTruthValues = coordX > left
    xEndTruthValues = coordX < right

    # merging boolean list to get truth table
    xTruthValues = xEntryTruthValues & xEndTruthValues
    yTruthValues = yEntryTruthValues & yEndTruthValues

    # get index of picture where the truth value is true
    resx = np.where(xTruthValues == True)
    resy = np.where(yTruthValues == True)

    return resx, resy

# converting dataset 
# observation data only
def prepareObservation(obs, grayscale=False):
    # loop through all available data
    firstData = True
    for i in range(len(obs)):
        # loop through dataset
        firstDataset = True
        for j in range(len(obs[i])):
            if j == 2 or j == 3:
                continue
            else :
                # check if grayscale or not
                if grayscale:
                    img = color.rgb2gray(color.rgba2rgb(obs[i][j]))
                    flat = img.reshape(obs[i][j].shape[0]*obs[i][j].shape[1])
                else:
                    img = obs[i][j]
                    flat = img.reshape(obs[i][j].shape[0]*obs[i][j].shape[1]*obs[i][j].shape[2])
                
                if firstDataset:
                    flattened = flat.copy()
                    firstDataset = False
                else :
                    flattened = np.hstack((flattened, flat))
        if firstData:
            data = flattened.copy()
            data = data.reshape(1, data.shape[0])
            firstData = False
        else :
            flattened = flattened.reshape(1, flattened.shape[0])
            data = np.vstack((data, flattened))
    return data

# prediction data only
def preparePrediction(pred, grayscale=False):
    # loop through all available data
    firstData = True
    for i in range(len(pred)):
        # loop through dataset
        firstDataset = True
        for j in range(len(pred[i])):
            if False:
                continue
            else :
                # check if grayscale or not
                if grayscale:
                    img = color.rgb2gray(color.rgba2rgb(pred[i][j]))
                    flat = img.reshape(pred[i][j].shape[0]*pred[i][j].shape[1])
                else:
                    img = pred[i][j]
                    flat = pred[i][j].reshape(pred[i][j].shape[0]*pred[i][j].shape[1]*pred[i][j].shape[2])
                
                
                if firstDataset:
                    flattened = flat.copy()
                    firstDataset = False
                else :
                    flattened = np.hstack((flattened, flat))
        if firstData:
            data = flattened.copy()
            data = data.reshape(1, data.shape[0])
            firstData = False
        else :
            flattened = flattened.reshape(1, flattened.shape[0])
            data = np.vstack((data, flattened))
    return data

# observation and prediction data
def prepareCombination(obs, pred, grayscale=False):
    # loop through all available data
    firstData = True
    for i in range(len(pred)):
        # loop through dataset
        firstDataset = True
        for j in range(len(pred[i])):
            # check if grayscale or not
            if grayscale:
                img = color.rgb2gray(color.rgba2rgb(pred[i][j]))
                flatP = img.reshape(pred[i][j].shape[0]*pred[i][j].shape[1])
            else:
                img = pred[i][j]
                flatP = img.reshape(pred[i][j].shape[0]*pred[i][j].shape[1]*pred[i][j].shape[2])
            
            obsCheck = j == 2 or j == 3
            if not obsCheck:
                # check if grayscale or not
                if grayscale:
                    img = color.rgb2gray(color.rgba2rgb(obs[i][j]))
                    flatO = img.reshape(obs[i][j].shape[0]*obs[i][j].shape[1])
                else:
                    img = obs[i][j]
                    flatO = img.reshape(obs[i][j].shape[0]*obs[i][j].shape[1]*obs[i][j].shape[2])
            
            if firstDataset:
                flattened = flatP.copy()
                if not obsCheck:
                    flattened = np.hstack((flattened, flatO))
                firstDataset = False
            else :
                flattened = np.hstack((flattened, flatP))
                if not obsCheck:
                    flattened = np.hstack((flattened, flatO))
                
        if firstData:
            data = flattened.copy()
            data = data.reshape(1, data.shape[0])
            firstData = False
        else :
            flattened = flattened.reshape(1, flattened.shape[0])
            data = np.vstack((data, flattened))
    return data

# Normalizing input data
def normalizingLabels(adta):
    '''
    Return normalized input data from 0 to 1, min, max value to convert back to predicted label
    '''
    minStat = np.min(adta)
    maxStat = np.max(adta)

    norm = (adta - minStat)/(maxStat - minStat)
    
    return norm, minStat, maxStat


def splitTrainTest(data, label, startBound=None, endBound=None, split=0.8, shuffle=False, randomSeed=None):
    
    if shuffle:
        random.seed(randomSeed)
        merge = list(zip(data, label))
        try:
            print(data.shape, label.shape)
        except Exception:
            pass
        random.shuffle(merge)
        data, label = zip(*merge)
        data = np.array(data)
        label = np.array(label)
        #random.shuffle(data)
        #random.shuffle(label)
    
    boundData = data[startBound:endBound]
    boundLabel = label[startBound:endBound]
    
    splitBound = round(split*len(boundLabel))
    trainData = boundData[:splitBound]
    trainLabel = boundLabel[:splitBound]
    testData = boundData[splitBound:]
    testLabel = boundLabel[splitBound:]
    
    return (trainData, trainLabel), (testData, testLabel)

def splitTrainTestSequential(data, label, startBound=None, endBound=None, split=0.8):
    return splitTrainTest(data, label, startBound, endBound, split)

def correctingWindData(datasetList=('winu', 'wn10', 'wind'), stdDimension=(400,100)):
    '''
    Check for read and dimension error in dataset
    Input datasetList : array like list of data
    '''
    p0 = os.path.join(os.path.dirname(os.getcwd()), 'mining_sadewa', 'sadewa')

    for i in datasetList:
        p1 = os.path.join(p0, i)
        p1r = os.path.join(p0, f'{i}_r')
        for j in os.listdir(p1):
            p2 = os.path.join(p1, j)
            p2r = os.path.join(p1r, j)
            try :
                img = mpimg.imread(p2)
                if img.shape != stdDimension:
                    print(f'Non standard dimensions at: {p2}. Fixed')
                    plt.imsave(p2r, prevImg)
                else:
                    # pass the read and dimension check
                    rz = skimage.transform.resize(img, stdDimension)
                    plt.imsave(p2r, rz)
                    prevImg = copy.deepcopy(rz)
            except Exception:
                print(f'Read error at: {p2}. Fixed.')
                plt.imsave(p2r, prevImg)
                      

def preparingSimulationData(usedDatas, hourOffsets=(0,), dimension=72, wlstation='manggarai'):
    '''
    Input :
    -- usedDatas : array like array
    -- hourOffsets : array like integer for costumizing manggarai date input data
    -- dimension : input data dimension (default : 72)
    -- !split : split slice between train/allavailabledata
    -- !shuffle : wether or not the x->y data pairs randomly shuffled or just sequence
    -- !randomSeed : random batch identification
    
    Returning dictionary of :
    -- fname : dataset name
    -- adta : available sliced input data between manggarai WL and sadewa
    -- adte : available sliced input date between manggarai WL and sadewa
    -- norm : normalized manggarai WL data
    -- minStat : minimum value of manggarai WL data
    -- maxStat : maximum value of manggarai WL data
    -- dataset : raw input data
    -- flattened : flattened raw input data
    -- traintest : (trainData, trainLabel), (testData, testLabel) tuple
    '''

    himawariData={}
    for hourOffset in hourOffsets:
        adte, adta = getAvailableSlicedData(dataScope='prediction', hourOffset=hourOffset, wlstation=wlstation)
        adta = np.array(adta).astype('float32')
        # normalizing input data
        norm, minStat, maxStat = normalizingLabels(adta)
        for usedData in usedDatas:
            # load data
            fname = ''.join(usedData) + dimension + hourOffset + .hdf5
            with h5py.File(os.path.join(os.getcwd(), 'dataset', fname), 'r') as f:
                data = f['datas'][()]

            flattened = preparePrediction(data, grayscale=True)
            
            himawariData[dictKey]={'fname':'{}'.format(dictKey),
                                    'hourOffset':hourOffset,
                                    'adta':adta,
                                    'adte':adte,
                                    'norm':norm,
                                    'minStat':minStat,
                                    'maxStat':maxStat,
                                    'dataset':data,
                                    'flattened':flattened,
                                    'traintest': splitTrainTest(flattened, norm, split=0.7, shuffle=True, randomSeed=10)}
    return himawariData

def generateRNNInput(adte, adta, recurrentCount=1):
    '''
    Check and return a tuple of date containing available data for recurrent configuration
    
    This is a sub-function to restack current cropped data into rnn enabled data based on recurrentCount number
    
    Return:
    recurrentIndexList = [(index-2, index-1, index+0), (index-1, index+0, index+1), (index-recurrentCount+index, index-recurrentCount+1+index, index-recurrentCount+2+index), ...]
    availableRecurrentDate = array like containing available date in recurrent configuration (in t=0)
    availableRecurrentLabel = array like containing available data label in recurrent configuration
    '''
    
    # defining start index
    # defining list to store the recurrent index
    recurrentIndexList = []
    availableRecurrentDate = []
    availableRecurrentLabel = []
    for idx in range(len(adte[recurrentCount:])):
        # check sequence
        checkSeq = [adte[idx+recurrentCount]+tdelta(hours=-recurrentCount)+tdelta(hours=x) for x in range(recurrentCount+1)]
        realSeq = [adte[idx+x] for x in range(recurrentCount+1)]
        if checkSeq != realSeq:
            continue
        else:
            recurrentIndexList.append([idx+x for x in range(recurrentCount+1)])
            availableRecurrentDate.append(adte[idx+recurrentCount])
            availableRecurrentLabel.append(adta[idx+recurrentCount])
    
    return recurrentIndexList, availableRecurrentDate, availableRecurrentLabel


def restackRNNInput(recurrentIndexList, dataset, flattened=False, grayscale=True):
    '''
    Create a new datasets in rnn mode by passing recurrentIndexList and dataset that want to be restacked
    
    Input:
    flattened : False(default)/True
    
    Output :
    restacked dataset (flattened / not flattened)
    '''
    firstData = True
    for sequences in recurrentIndexList:
        first = True
        for sequence in sequences:
            if first:
                stacked = copy.deepcopy(dataset[sequence])
                first = False
            else:
                stacked = np.vstack((stacked, dataset[sequence]))
        # reshape stacked data
        stacked = stacked.reshape(1, stacked.shape[0], stacked.shape[1], stacked.shape[2], stacked.shape[3])
        if firstData:
            allStacked = copy.deepcopy(stacked)
            firstData = False
        else:
            allStacked = np.vstack((allStacked, stacked))
    
    if flattened:
        print(allStacked.shape)
        return preparePrediction(allStacked, grayscale=grayscale)
    else:
        return allStacked

    
def performRNNDatasetCreation(usedDatas, dims, recurrentLists, dataScope='prediction', wlstation='manggarai', flattened=True):
    '''
    Performing RNN Data Creation by passing data combination that want to be recreated as RNN sequence and list of number that acting as
    how much sequence that want to be added before the t+0 data. For ex if the recurrentLists[0] says 2, it means that there will be 3 stacked data,
    t-2, t-1, t+0.
    
    This function can process from 1 to 6 data combination(s)
    
    Input:
    -usedDatas : array like of array of data combination(s) (up to 6) in sequence with dims
    -dims : array like of dimensions, in squence with usedDatas
    -recurrentLists : array like of lists of number that acting as how much sequence that want to be added before the t+0 data (>=1)
    
    '''
    rls = recurrentLists

    adte, adta = getAvailableSlicedData(maxData=True, hourOffset=0, dataScope=dataScope, wlstation=wlstation)
    rils = []
    for rl in rls:
        ril, *_ = generateRNNInput(adte, adta, recurrentCount=rl)
        rils.append(ril)

    for j in range(len(usedDatas)):
        usedData = usedDatas[j]
        dim = dims[j]

        ffn = ''.join(usedData) + dim + '.hdf5'
        fpath = os.path.join(os.getcwd(), 'dataset', 'manggaraiRNN', ffn)

        with h5py.File(fpath,'r') as f:
            data = f['datas'][()]

        for i, v in enumerate(recurrentLists):
            print('{}-{}-{}'.format(fileName, dim, v))
            # restacking the data
            allStacked = restackRNNInput(rils[i], data, flattened=flattened)
            
            # save restacked data to file
            fpsave = os.path.join(os.getcwd(), 'dataset', 'manggaraiRNN', ffn.replace('.hdf5', f'r{v}f.hdf5'))
            with h5py.File(fpsave, 'w') as f:
                f.create_dataset('datas', data=allStacked)


