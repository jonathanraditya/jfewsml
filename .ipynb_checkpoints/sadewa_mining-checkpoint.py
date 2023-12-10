import requests
import datetime
import time
import os

# defining function to download the data

def executeSadewa(data, sleepTime, inputStartDate=False, startDate=None):
    '''
    Execute sadewa download for specific index
    '''
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
        },
    }
    
    # set or load initial loop
    try:
        with open('{}.txt'.format(data), 'r') as f:
            initLoop=f.read()
    except:
        initLoop=0
        with open('{}.txt'.format(data), 'w') as f:
            f.write(str(initLoop))
            
    os.makedirs('./sadewa/{}'.format(data), exists_ok=True)
    
    today=datetime.datetime.now()
    yearStart=database[data]['yearStart']
    if inputStartDate :
        startDate=datetime.datetime(*startDate)
    else:
        startDate=datetime.datetime(int(yearStart), 1, 1)
    
    # offsetting start date if the loop have been started before
    if int(initLoop) > 0:
        startDate=startDate+datetime.timedelta(int(initLoop))
    dateRange=(today-startDate).days
    
    
    
    for i in range(dateRange+1):
        dateLoop=(startDate+datetime.timedelta(i))
        
        # loop from 0 to 23 
        hours=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
        for hour in hours:
            # building URL
            burl=database[data]['url']
            fname=database[data]['fname']
            furl='{}{}'.format(burl, fname)
            url=furl.format(dateLoop.strftime('%Y'), dateLoop.strftime('%m'), dateLoop.strftime('%d'),
                           dateLoop.strftime('%Y'), dateLoop.strftime('%m'), dateLoop.strftime('%d'), hour)
            print('{}-{} Current URL : {}'.format(int(initLoop)+1+i, hour, url))
            
            # try to fetch data from the server 
            try:
                response=requests.get(url)
            
            # except error occured during fetching, continue to next iteration after error logging to file
            except:
                # save error log to file for future trying
                with open('sadewaerr.txt', 'a') as errlog:
                    errlog.write(url)
                    
                # sleep for 10 seconds
                time.sleep(10)
                
                # continue to next iteration
                continue
                
                
            # if server responded but bringing >=400 status code, continue to next iteration after error logging to file
            if not response.ok:
                # save error log to file for future trying
                with open('sadewaerr.txt', 'a') as errlog:
                    errlog.write(url)
                    
                # sleep for 10 seconds
                time.sleep(10)
                
                # continue to next iteration
                continue
            else:
                # save response to file
                formatfname=fname.format(dateLoop.strftime('%Y'), dateLoop.strftime('%m'), dateLoop.strftime('%d'), hour)
                with open('./sadewa/{}/{}'.format(data, formatfname), 'wb') as fsave:
                    fsave.write(response.content)
                    
            # perform sleep between loop
            time.sleep(sleepTime)

        # update current loop to file
        with open('{}.txt'.format(data), 'w') as initf:
            initf.write(str(i+int(initLoop)))

        
        # perform sleep between loop
        time.sleep(sleepTime)


# executeSadewa(data, sleepTime, inputStartDate=False, startDate)
executeSadewa(input('Enter Data type :'), int(input('How much time reserved for sleep?')))