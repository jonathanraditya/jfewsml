# import modules
import sqlite3
from sqlite3 import Error
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
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


def hello():
    print('Hello!')

def showSampleSadewaData():
    '''
    showing sadewa sampel data 
    current : cloud-0
    '''
    himawariPath='../mining_sadewa/sadewa/'
    himawari=getHimawariFilename()

    data=himawari['cloud'][0]
    fullPath='{}{}/{}'.format(himawariPath, 'cloud', data)
    img=mpimg.imread(fullPath)
    plt.imshow(img, cmap='rainbow')
    plt.colorbar()

def create_connection(db_file):
    '''
    create a database connection to a SQLite database
    specified by db_file
    :param db_file : database file
    :return: Connection Object or None
    '''
    conn=None
    try:
        conn=sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)  
    
def printSadewaFetchMissingData():
    '''
    printing total count of missing data for every sadewa-himawari dataset
    '''
    sadPath='../mining_sadewa/sadewaerr.txt'
    with open(sadPath, 'r') as sErrFile:
        errContent=sErrFile.read()

    rawErrList=errContent.split('.png')
    cleanErrUrl=[]
    for err in rawErrList:
        cleanErrUrl.append('{}.png'.format(err))

    database={
            'IR1':{
                'url':'https://sadewa.sains.lapan.go.id/HIMAWARI/himawari_merc/IR1/{}/{}/{}/',
                'fname':'H89_IR1_{}{}{}{}00.png',
                'yearStart':'2020'
            },
            'IR3':{
                'url':'https://sadewa.sains.lapan.go.id/HIMAWARI/himawari_merc/IR3/{}/{}/{}/',
                'fname':'H89_IR3_{}{}{}{}00.png',
                'yearStart':'2020'
            },
            'VIS':{
                'url':'https://sadewa.sains.lapan.go.id/HIMAWARI/himawari_merc/VIS/{}/{}/{}/',
                'fname':'H89_VIS_{}{}{}{}00.png',
                'yearStart':'2020'
            },
            'B04':{
                'url':'https://sadewa.sains.lapan.go.id/HIMAWARI/himawari_merc/B04/{}/{}/{}/',
                'fname':'H89_B04_{}{}{}{}00.png',
                'yearStart':'2020'
            },
            'CCLD':{
                'url':'https://sadewa.sains.lapan.go.id/HIMAWARI/komposit/{}/{}/{}/',
                'fname':'H89_CCLD_{}{}{}{}00.png',
                'yearStart':'2020'
            },
            'rain':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'rain_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'cloud':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'cloud_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'psf':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'psf_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'qvapor':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'qvapor_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'sst':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'sst_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'wind':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'wind_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'winu':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'winu_{}{}{}_{}.png',
                'yearStart':'2019'
            },
            'wn10':{
                'url':'https://sadewa.sains.lapan.go.id/wrf/{}/{}/{}/',
                'fname':'wn10_{}{}{}_{}.png',
                'yearStart':'2019'
            }
        }

    # making list of data keys through list comprehension
    dataKeys=[x for x in database.keys()]
    errClassification={}
    # establishing dictionary
    for key in dataKeys:
        errClassification[key]=[]

    # loop to append error URL for each data type
    for err in cleanErrUrl:
        # loop for each key to check for True value
        for key in dataKeys:
            # check for match data type
            if key in err:
                errClassification[key].append(err)
                continue

    # print count of missing data for each data 
    for errKey in errClassification.keys():
        print(errKey, len(errClassification[errKey]))
        
def idealDataCount():
    '''
    returning (2019idc, 2020idc) ideal data count for Himawari dataset
    '''
    # calculating ideal data count for each entry date
    early2019=[2019,1,1]
    early2020=[2020,1,1]
    minedDate=[2021,3,14]
    fdE19=datetime.datetime(*(early2019))
    fdE20=datetime.datetime(*(early2020))
    fdMD=datetime.datetime(*(minedDate))

    dateRange2019=(fdMD-fdE19).days
    dateRange2020=(fdMD-fdE20).days

    # ideal data count for each entry date
    dataCount2019=dateRange2019*24
    dataCount2020=dateRange2020*24
    
    return dataCount2019, dataCount2020

def manggaraiFullData():
    # read and fetch database data to pandas dataframe
    dsdaPath='../mining_dsda/dsda.db'
    conn=create_connection(dsdaPath)
    manggarai=pd.read_sql_query('SELECT * FROM manggarai', conn)

    # set main index to currentdate
    manggarai.set_index('currentdate')

    # convert data type from object to string
    manggaraiConv=manggarai.convert_dtypes()

    # set main index to currentdate
    manggaraiConv.set_index('currentdate')

    # convert date datatype to datetime64[ns]
    manggaraiConv['currentdate']=manggaraiConv['currentdate'].astype('datetime64[ns]')
    
    return manggaraiConv

def manggaraiDataList(maxData=True, hourOffset=0, wlstation='manggarai'):
    '''
    Returning a tuple of list (date, data) of manggarai TMA data with 10-minutes-interval from DSDA dataset in year 2020
    '''
    # read and fetch database data to pandas dataframe
    dsdaPath='../mining_dsda/dsda.db'
    conn=create_connection(dsdaPath)
    manggarai=pd.read_sql_query('SELECT * FROM {}'.format(wlstation), conn)

    # set main index to currentdate
    manggarai.set_index('currentdate')

    # convert data type from object to string
    manggaraiConv=manggarai.convert_dtypes()

    # set main index to currentdate
    manggaraiConv.set_index('currentdate')

    # convert date datatype to datetime64[ns]
    manggaraiConv['currentdate']=manggaraiConv['currentdate'].astype('datetime64[ns]')

    # slicing data to 2020 timeframe
    #mask = (manggaraiConv['currentdate'] >= '2019-02-01 00:00') & (manggaraiConv['currentdate'] <= '2021-04-03 23:50')
    mask = (manggaraiConv['currentdate'] >= '2019-02-01 00:00')
    manggaraiSlice2020=manggaraiConv.loc[mask]

    # converting 10-minute-data to hourly data
    startDate=datetime.datetime(2019,2,1)
    minutes=[x*10 for x in range(6)]
    hours=[x for x in range(24)]
    days=[x for x in range(780)]

    dateListHourly=[]
    dataListHourly=[]
    for day in days:
        for hour in hours:
            hourlyData=[]

            # set error indicator back to false
            error=False

            for minute in minutes:
                # perform data fetch, add to list, and get max value
                dateLoop=startDate+datetime.timedelta(days=day, hours=hour+hourOffset, minutes=minute)
                rowFetch=manggaraiSlice2020.loc[(manggaraiSlice2020['currentdate'] == dateLoop)]
                #print(rowFetch)

                # try to fetch if the result is not zero
                try:
                    dataFetch=rowFetch['data'].item()
                    hourlyData.append(dataFetch)
                except ValueError:
                    error=True

            # insert data if error indicator is False
            if not error:
                # make hourly date using timedelta
                hourlyDate=startDate+datetime.timedelta(days=day, hours=hour)
                
                if maxData:
                    # get maximum value of hourly data
                    maxDataHourly=max(hourlyData)
                else:
                    # get maximum value of hourly data
                    maxDataHourly=hourlyData.mean()

                # insert value to global list
                dateListHourly.append(hourlyDate)
                dataListHourly.append(maxDataHourly)
            else: # if error occured during data fetch (null or something else)
                continue # to next loop
    return dateListHourly, dataListHourly

def getHimawariFilename():
    '''
    Return dictionary of available himawari data based on filename inside
    folder as a key
    '''
    himawariPath='../mining_sadewa/sadewa/'
    # load folder name
    directory=[directory for directory in os.listdir(himawariPath)]

    # store fileame
    himawari={}

    # load all filename stored on disk to dictionary with each folder name as keys
    for direct in directory:
        fpath='{}{}'.format(himawariPath, direct)
        himawari[direct]=[fname for fname in os.listdir(fpath)]
        
    return himawari

def extractHimawariDatetime():
    '''
    Extract every filename in sadewa-himawari data to datetime object for easier handling
    
    Returns :
    extractedDate -- dictionary containing list of datetime object for each filename inside dictionary keys for every data
    '''
    himawari=getHimawariFilename()

    # extract date for each himawari data type to datetime.datetime object
    observations=['CCLD','B04','IR1','IR3','VIS']
    extractedDate={}
    for obs in observations:
        extractedDate[obs]=[datetime.datetime.strptime(x.replace('H89_{}_'.format(obs),'').replace('.png',''), '%Y%m%d%H%M') for x in himawari[obs]]

    predictions=['cloud','psf','qvapor','rain','sst','wind','winu','wn10']
    for pred in predictions:
        extractedDate[pred]=[datetime.datetime.strptime(x.replace('{}_'.format(pred),'').replace('.png','').replace('_','')+'00', '%Y%m%d%H%M') for x in himawari[pred]]
        
    return extractedDate

def getAvailableSlicedData(maxData=True, hourOffset=0, dataScope='combination', wlstation='manggarai'):
    '''
    check through all available dataset, including manggarai TMA, sadewa-himawari IR1, IR3, VIS, B04, and CCLD
    and return a tuple containing datetime object and manggarai hourly TMA data that are synced through all available dataset
    
    This function doesn't return sadewa-himawari data, because using the datetime format and the sadewa-himawari data types,
    the full name of the file required can be constructed.
    
    return : (slicedDate, slicedData) # both are lists inside a tuple
    '''
    extractedDate = extractHimawariDatetime()
        
    # getting date-data slice from himawari and manggarai TMA data

    # using function to get manggarai available date-data
    dateListHourly, dataListHourly = manggaraiDataList(maxData, hourOffset, wlstation=wlstation)

    # loop to every data
    # check algorithm : manggarai checked against every himawari data, and if all true, date is inserted to sliced data
    slicedDate=[]
    slicedData=[]
    for i in range(len(dateListHourly)):
        
        if dataScope == 'combination':
            usedData=['CCLD','B04','IR1','IR3','VIS','rain','cloud','psf','qvapor','sst']
        elif dataScope == 'prediction':
            usedData=('cloud','psf','qvapor','rain','sst','wind','winu','wn10')

        # defining control mechanism
        checked=True

        # loop through every himawari data
        for used in usedData:
            if dateListHourly[i] not in extractedDate[used]:
                checked=False # set checked to False if there are no complementary data found in another dataset

        # input data if all checked
        if checked:
            slicedDate.append(dateListHourly[i])
            slicedData.append(dataListHourly[i])
    return slicedDate, slicedData


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
    imgPath = '../mining_sadewa/sadewa/{}/{}_{}_{}.png'

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
    imgPath = '../mining_sadewa/sadewa/{}/H89_{}_{}.png'

    return cropImageData(imgCropX, imgCropY, adte, usedDatas, imgPath)

def performIndividualCropPredictionData(dim):
    '''
    Perform individual database creation of prediction data
    '''
    # initializing individual variables
    usedDatas = [['cloud'],['psf'],['qvapor'],['rain'],['sst'],['wind'],['winu'],['wn10']]
    for usedData in usedDatas:
        
        crop = cropImagePredictionData(dim, usedDatas=usedData)
        with h5py.File('dataset/{}{}.hdf5'.format(usedData[0], dim), 'w') as f:
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
            imgPath='../mining_sadewa/sadewa/{}/H89_{}_{}.png'.format(data,data, datef)
            image=color.rgb2gray(color.rgba2rgb(mpimg.imread(imgPath)))
            #image=color.rgba2rgb(mpimg.imread(imgPath))
            image=image[imgCropX[0]:imgCropX[1], imgCropY[0]:imgCropY[1]]
            # crop image
            reshapedP=image.reshape(image.shape[0]*image.shape[1])
            reshaped=np.append(reshaped, reshapedP)
            #plt.imshow(image, cmap='gray')
        # transpose image
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

def sigmoid(z):
    '''
    Compute the sigmoid of z
    
    Arguments :
    z -- A scalar or numpy array of any size
    
    Return :
    s -- sigmoid(z)
    '''
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    '''
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0
    
    Argument :
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns :
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    '''
    
    w = np.zeros((dim, 1))
    b = 0
    
    return w,b

def propagate(w, b, X, Y):
    '''
    Implement the cost function and it's gradient fot the propagation
    
    Arguments :
    w -- weights, a numpy array of size (num_px*num_px*num_channels, 1)
    b -- bias, a scalar
    X -- data of size (num_px*num_px*num_channels, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    
    Return :
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    '''
    # number of training examples
    m = X.shape[1]
    
    # forward propagation
    # w.T to make sure the shapes is aligned to create dot product (a,b) dot (b,c) dimensions
    # shape of A is (1, m train ex)
    # computing activation
    A = sigmoid(np.dot(w.T, X)+b)
    
    # sum over m training examples
    # compute cost
    cost = -1/m*(np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))
    cost = np.squeeze(cost)
    
    # backward propagation
    # be careful for the placement of X and transpose over substraction of A with Y
    # because the required dimension(s) are (nodes, m) dot (m, 1) -> (nodes, 1)
    dw = 1/m*(np.dot(X, ((A-Y).T)))
    # sum over m training examples after substraction
    db = 1/m*(np.sum(A-Y))
    
    grads = {'dw':dw, # dw shapes : (nodex, 1)
             'db':db} # db is a float number, not a matrix
    return grads, cost

def optimize(w,b,X,Y, num_iterations, learning_rate, print_cost=True):
    '''
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments :
    w -- weights, a numpy array of size(num_px*num_px*num_channel,1)
    b -- bias, a scalar
    X -- data of shape (num_px*num_px*num_channel, number of examples)
    Y -- label vector of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns :
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    '''
    costs = []
    
    for i in range(num_iterations):
        # cost and gradient calculation
        grads, cost = propagate(w,b,X,Y)
        
        # retrieve derivatives from grads
        dw = grads['dw']
        db = grads['db']
        
        # update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # record the costs
        if i % 100 == 0:
            costs.append(cost)
            
        # print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print('Cost after iteration {} : {}'.format(i, cost))
    
    params = {'w':w,
              'b':b}
    
    grads = {'dw':dw,
             'db':db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict wether the label using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_pc*num_px*num_channel, 1)
    b -- bias, a scalar
    X -- data of size (num_px*num_px*num_channel, number of examples)
    
    Returns :
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    #Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    # compute vector 'A' predicting y_hat value
    A = sigmoid(np.dot(w.T, X) + b)
    
    return A

def executeModel(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    '''
    Builds the logistic regression model by calling component functions
    
    Arguments :
    X_train -- training set represented by a numpy array of shape (num_px*num_px*num_channel, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px*num_px*num_channel, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- set to true to print the cost every 100 iterations
    
    Returns :
    d -- dictionary containing information about the model
    '''
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # retrieve parameters w and b from dictionary 'parameters'
    w = parameters['w']
    b = parameters['b']
    
    # predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # print train/test Errors
    #trainAcc = (100 - np.mean(np.abs(Y_prediction_train - Y_train))) * 100
    #testAcc = (100 - np.mean(np.abs(Y_prediction_test - Y_test))) * 100
    trainAcc = (1-np.mean(np.abs(Y_prediction_train - Y_train)))*100
    testAcc = (1-np.mean(np.abs(Y_prediction_test - Y_test)))*100
    print('Train accuracy : {} %'.format(trainAcc))
    print('Test accuracy : {} %'.format(testAcc))
    
    d = {'costs':costs,
         'Y_prediction_test':Y_prediction_test,
         'Y_prediction_train':Y_prediction_train,
         'w':w,
         'b':b,
         'learning_rate':learning_rate,
         'num_iterations':num_iterations}
    
    return d

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

# resize function for wind data

def correctingWindData():
    dataset = ('winu', 'wn10', 'wind')
    paths = {}
    for data in dataset:
        paths[data] = os.listdir(folderPath.format(data))

    for path in paths:
        print('Processing {} data'.format(path))
        for filename in paths[path]:
            # check if readable
            if filename in readError:
                # use previous data
                plt.imsave('../mining_sadewa/sadewa/{}_r/{}'.format(path, filename), prevImg)
            # check if in correct dimension
            elif filename in nonStdDim:
                # use previous data
                plt.imsave('../mining_sadewa/sadewa/{}_r/{}'.format(path, filename), prevImg)
            else:
                img = mpimg.imread('../mining_sadewa/sadewa/{}/{}'.format(path, filename))

                # resize image to correct dimension(s)
                resized = skimage.transform.resize(img, (400,1000))
                plt.imsave('../mining_sadewa/sadewa/{}_r/{}'.format(path, filename), resized)

                prevImg = copy.deepcopy(resized)

def checkDataError(datasetList, stdDimension):
    '''
    Check for read and dimension error in dataset
    Input datasetList : array like list of data
    stdDimension : a tuple containing standard dimension (and color channel(s)) of image
    Returning 2 list : readError and nonStdDim
    '''
    paths = {}
    for data in dataset:
        paths[data] = os.listdir(folderPath.format(data))

    # read test
    readError = []
    nonStdDim = []
    for path in paths:
        for filename in paths[path]:
            try :
                img = mpimg.imread('../mining_sadewa/sadewa/{}/{}'.format(path, filename))
                if img.shape != stdDimension:
                    print('Non standard dimensions : {}'.format(filename))
                    nonStdDim.append(filename)
            except Exception:
                print('Error occured : {}'.format(filename))
                readError.append(filename)

    return readError, nonStdDim

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
            inputCombination = len(usedData)
            if inputCombination == 1:
                dictKey = '{}{}-{}'.format(usedData[0], dimension, hourOffset)
                fname = 'dataset/{}.hdf5'.format(dictKey)
            elif inputCombination == 2:
                dictKey = '{}{}{}-{}'.format(usedData[0], usedData[1], dimension, hourOffset)
                fname = 'dataset/{}.hdf5'.format(dictKey)
            elif inputCombination == 3:
                dictKey = '{}{}{}{}-{}'.format(usedData[0], usedData[1], usedData[2], dimension, hourOffset)
                fname = 'dataset/{}.hdf5'.format(dictKey)
            elif inputCombination == 4:
                dictKey = '{}{}{}{}{}-{}'.format(usedData[0], usedData[1], usedData[2], usedData[3],dimension, hourOffset)
                fname = 'dataset/{}.hdf5'.format(dictKey)
                
                

            with h5py.File(fname, 'r') as f:
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
        checkSeq = [adte[idx+recurrentCount]+datetime.timedelta(hours=-recurrentCount)+datetime.timedelta(hours=x) for x in range(recurrentCount+1)]
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

    adte, adta = getAvailableSlicedData(maxData=True, hourOffset=0, dataScope=dataScope, wlstation=wlstation)
    recurrentIndexLists=[]
    for recurrentList in recurrentLists:
        recurrentIndexList, availableRecurrentDate, availableRecurrentLabel = generateRNNInput(adte, adta, recurrentCount=recurrentList)
        recurrentIndexLists.append(recurrentIndexList)

    for j in range(len(usedDatas)):
        usedData = usedDatas[j]
        dim = dims[j]
        # define the length of data
        dataLength = len(usedData)
        # read stored data
        if dataLength == 1:
            fileName = '{}{}'.format(usedData[0], dim)
        elif dataLength == 2:
            fileName = '{}{}{}'.format(usedData[0], usedData[1], dim)
        elif dataLength == 3:
            fileName = '{}{}{}{}'.format(usedData[0], usedData[1], usedData[2], dim)
        elif dataLength == 4:
            fileName = '{}{}{}{}{}'.format(usedData[0], usedData[1], usedData[2], usedData[3], dim)
        elif dataLength == 5:
            fileName = '{}{}{}{}{}{}'.format(usedData[0], usedData[1], usedData[2], usedData[3], usedData[4], dim)
        elif dataLength == 6:
            fileName = '{}{}{}{}{}{}{}'.format(usedData[0], usedData[1], usedData[2], usedData[3], usedData[4], usedData[5], dim)
        print(fileName)
        fpath = 'dataset/manggaraiRNN/{}.hdf5'.format(fileName)
        with h5py.File(fpath,'r') as f:
            data = f['datas'][()]
            
        for i in range(len(recurrentLists)):
            print('{}-{}-{}'.format(fileName, dim, recurrentLists[i]))
            # restacking the data
            allStacked = restackRNNInput(recurrentIndexLists[i], data, flattened=flattened)
            
            # save restacked data to file
            with h5py.File('dataset/manggaraiRNN/{}r{}f.hdf5'.format(fileName, recurrentLists[i]), 'w') as f:
                f.create_dataset('datas', data=allStacked)


