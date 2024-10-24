import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import datetime as dt
import gpxpy
from tqdm import tqdm
from urllib.error import URLError
from urllib.request import Request, urlopen
import time
import os
import shutil
import matplotlib as mpl
import math
import cv2





USE_CACHED=True#whether to use our cached dataframe

CACHED_ACTIVITIES_FILE="extractedData.pickle"#file for cached dataframe
INPUT_DIR='processed_files'#if not using cached dataframe, directory for gpx files
TILE_DIR="tiles"#directory to store map tiles
FRAME_DIR="frames"

OSM_TILE_SERVER = 'https://tile.openstreetmap.org/{}/{}/{}.png' # OSM tile url from https://wiki.openstreetmap.org/wiki/Raster_tile_providers
OSM_TILE_SIZE = 256 # Tile size of the tiles we download from openstreetmap

MIN_START_LAT=41.31
MAX_START_LAT=41.32
MIN_START_LON=-72.93
MAX_START_LON=-72.92
MIN_DATE=dt.datetime(2022,12,31).date()


ZOOM_LEVEL=14

DARK_MAP=True
SIGMA=3#size of the trail
HEATMAP=False
MARGIN_SIZE=32#in pixels
PLT_COLORMAP = 'plasma' # matplotlib color map
SHOW_LEAD=True
SHOW_MILEAGE=True

NUM_FRAMES=2000


def deg2xy(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Returns OSM coordinates (x,y) from (lat,lon) in degree"""
    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    lat_rad = np.radians(lat_deg)
    n = 2.0**zoom
    x = (lon_deg+180.0)/360.0*n
    y = (1.0-np.arcsinh(np.tan(lat_rad))/np.pi)/2.0*n
    return x, y

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Returns image filtered with a gaussian function of variance sigma**2"""
    i, j = np.meshgrid(np.arange(image.shape[0]),
                       np.arange(image.shape[1]),
                       indexing='ij')
    mu = (int(image.shape[0]/2.0),
          int(image.shape[1]/2.0))
    gaussian = 1.0/(2.0*np.pi*sigma*sigma)*np.exp(-0.5*(((i-mu[0])/sigma)**2+\
                                                        ((j-mu[1])/sigma)**2))
    gaussian = np.roll(gaussian, (-mu[0], -mu[1]), axis=(0, 1))
    image_fft = np.fft.rfft2(image)
    gaussian_fft = np.fft.rfft2(gaussian)
    image = np.fft.irfft2(image_fft*gaussian_fft)
    return image


def distance(lat1, lon1, lat2, lon2):
    #r = 6371 # km
    r = 3958.8 #miles
    p = np.pi / 180

    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 2 * r * np.arcsin(np.sqrt(a))



#Try to load the dataframe from cache
if USE_CACHED:

    try:
        print("Loading cached data from",CACHED_ACTIVITIES_FILE)
        activities = pickle.load(open(CACHED_ACTIVITIES_FILE, "rb"))
    except Exception:
        USE_CACHED=False
        print("Loading cached activities failed")
    else: 
        print(len(activities.startTime.unique())," valid activities found.")


def processActivity(gpx):

    startLat=None
    startLon=None
    startTime=None

    lats=[]
    lons=[]
    datetimes=[]
    runTime=[]

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                datetime=point.time
                if startLat is None:
                    startLat=point.latitude
                    startLon=point.longitude
                    startTime=datetime

                    if startTime.date()<MIN_DATE:
                        
                        return True,"out of date"
                        
                    
                    if startLat is not None and startLon is not None and startLat>MIN_START_LAT and startLat<MAX_START_LAT and startLon>MIN_START_LON and startLon<MAX_START_LON:
                        pass#it's in range
                    else:
                        return True,"out of range"


                lats.append(point.latitude)
                lons.append(point.longitude)
                datetimes.append(datetime)
                runTime.append(datetime-startTime)

    if(len(datetimes)==0):
        return True,"empty"

    return False,pd.DataFrame(data={'datetime':datetimes,'lat':lats,'lon':lons,"runTime":runTime,"startTime":[startTime]*len(lats),"startLat":[startLat]*len(lats),"startLon":[startLon]*len(lats),"totalTime":[max(runTime)]*len(lats)})

#Build dataframe
if not USE_CACHED:
    files=glob.glob(INPUT_DIR+"/*.gpx")


    nError=0
    nOutOfDate=0
    nOutOfRange=0
    nEmpty=0

    activities=[]


    for file in tqdm(files[:], desc="processing gpx files"):
        gpx_file = open(file, 'r')

        try:
            gpx = gpxpy.parse(gpx_file)
        except Exception:
            print(file,"could not be parsed by gpxpy")
            nError+=1
            continue

        

        hasErr,data=processActivity(gpx)
        if hasErr:
            if data=="out of range":nOutOfRange+=1
            elif data=="out of date":nOutOfDate+=1
            elif data=="empty":nEmpty+=1
        else:activities.append(data)

    activities=pd.concat(activities,axis=0)

    print(len(activities.startTime.unique())," valid activities found.",nError,"activities had an error.",nOutOfDate,"activities were out of date range.",nOutOfRange,"activities were out of location range.",nEmpty,"activities had empty gpx data.")
    print("Caching activity dataframe at ",CACHED_ACTIVITIES_FILE)
    pickle.dump(activities, file = open(CACHED_ACTIVITIES_FILE, "wb"))
    

#GET MAP TILES######################################################
#GET MAP TILES######################################################
#GET MAP TILES######################################################


overall_lat_max=activities.lat.max()
overall_lat_min=activities.lat.min()
overall_lon_max=activities.lon.max()
overall_lon_min=activities.lon.min()

x_tile_min, y_tile_max = map(int, deg2xy(overall_lat_min, overall_lon_min, ZOOM_LEVEL))
x_tile_max, y_tile_min = map(int, deg2xy(overall_lat_max, overall_lon_max, ZOOM_LEVEL))

tile_count = (x_tile_max-x_tile_min+1)*(y_tile_max-y_tile_min+1)

# download tiles
os.makedirs(TILE_DIR, exist_ok=True)

supertile = np.zeros(((y_tile_max-y_tile_min+1)*OSM_TILE_SIZE,
                        (x_tile_max-x_tile_min+1)*OSM_TILE_SIZE, 3))

n = 0
for x in range(x_tile_min, x_tile_max+1):
    for y in range(y_tile_min, y_tile_max+1):
        n += 1

        tile_file = 'tiles/tile_{}_{}_{}.png'.format(ZOOM_LEVEL, x, y)

        if not glob.glob(tile_file):
            print('downloading tile {}/{}'.format(n, tile_count))

            url = OSM_TILE_SERVER.format(ZOOM_LEVEL, x, y)

            request = Request(url, headers={'User-Agent': 'Strava-local-heatmap/master'})

            try:
                with urlopen(request, timeout=1) as response:
                    data = response.read()

                with open(tile_file, 'wb') as file:
                    file.write(data)

                tile = plt.imread(tile_file)

            except URLError as e:
                print('ERROR downloading failed, using blank tile: {}'.format(e))

                tile = np.ones((OSM_TILE_SIZE,
                                OSM_TILE_SIZE, 3))

            finally:
                time.sleep(1)

        else:
            #print('reading local tile {}/{}'.format(n, tile_count))

            tile = plt.imread(tile_file)

        i = y-y_tile_min
        j = x-x_tile_min

        supertile[i*OSM_TILE_SIZE:(i+1)*OSM_TILE_SIZE,
                    j*OSM_TILE_SIZE:(j+1)*OSM_TILE_SIZE, :] = tile[:, :, :3]
        
if(DARK_MAP):
    supertile = np.sum(supertile*[0.2126, 0.7152, 0.0722], axis=2) # to grayscale
    supertile = 1.0-supertile # invert colors
    supertile = np.dstack((supertile, supertile, supertile)) # to rgb

mapImg=supertile
del supertile


###########################################################################
#Generate data 

print(activities.columns)



#Show activites by percent complete (they all finish start and finish at the same time)
activities["percComplete"]=(activities["datetime"]-activities["startTime"])/activities["totalTime"]

#All activities start at the same time and are shown in sped-up real time
activities["realTime"]=(activities["datetime"]-activities["startTime"])/activities["totalTime"].max()

#Activites are shown sequential
activities["index"]=activities.startTime.rank(method="dense")#orders activities by time they started
activities["sequential"]=activities["index"]/activities["index"].max()+activities.percComplete*1/activities["index"].max()

#Activities are shown sequential based off of their angle
activities['meanLatOffset']=activities.groupby("startTime").lat.transform("mean")-activities.startLat.mean()
activities['meanLonOffset']=activities.groupby("startTime").lon.transform("mean")-activities.startLon.mean()
activities["angle"]=(2*np.pi-np.arctan2(activities.meanLonOffset,activities.meanLatOffset)+0.86*np.pi).mod(2*np.pi)
activities["angleRank"]=activities.angle.rank(method="dense")
activities["angleSequential"]=activities.angleRank/activities.angleRank.max()+activities.percComplete*1/activities["angleRank"].max()



activities["showPoint"]=activities["angleSequential"]*0.9




#Calculate distance total, ordered by our showPoint
activities=activities.sort_values("datetime")
shiftLat=activities.lat.shift()
shiftLon=activities.lon.shift()
activities["newDist"]=distance(shiftLat,shiftLon,activities.lat,activities.lon)
activities=activities.sort_values("showPoint")
activities["distTotal"]=np.cumsum(activities.newDist)
activities.loc[activities.distTotal.isna(),"distTotal"]=0


#convert x y data into pixels and get min and max so we can do image cropping later
overallXyData = np.array(deg2xy(activities.lat, activities.lon, ZOOM_LEVEL)).T
overallXyData = np.round((overallXyData-[x_tile_min, y_tile_min])*OSM_TILE_SIZE)
overallIjData=np.flip(overallXyData.astype(int), axis=1)
i_min, j_min = np.min(overallIjData, axis=0)
i_max, j_max = np.max(overallIjData, axis=0)


shutil.rmtree(FRAME_DIR,ignore_errors=True)
os.makedirs(FRAME_DIR,exist_ok=True)
for frameI,progress in tqdm(list(enumerate(np.linspace(0,1,NUM_FRAMES))),desc="Generating Frames"):
    df=activities[activities.showPoint<=progress]
    frame=mapImg.copy()

    data = np.zeros(mapImg.shape[:2])
    xy_data = deg2xy(df.lat, df.lon, ZOOM_LEVEL)

    xy_data = np.array(xy_data).T
    xy_data = np.round((xy_data-[x_tile_min, y_tile_min])*OSM_TILE_SIZE)
    ij_data = np.flip(xy_data.astype(int), axis=1) # to supertile coordinates

    if HEATMAP or not SHOW_LEAD:
        for i, j in ij_data:
            data[i-SIGMA:i+SIGMA, j-SIGMA:j+SIGMA] += 1.0
    else:
        for (i, j), showPoint in zip(ij_data,df.showPoint):

            timeSinceAppeared=(progress-showPoint)

            colorVal=1-min(timeSinceAppeared*40,0.90)
            #colorVal=min(timeSinceAppeared*10,1)

            area=data[i-SIGMA:i+SIGMA, j-SIGMA:j+SIGMA]
            area[area<colorVal]=colorVal
            data[i-SIGMA:i+SIGMA, j-SIGMA:j+SIGMA]=area


    

    if HEATMAP:
        #pixel resolution
        res_pixel = 156543.03*np.cos(np.radians(np.mean(activities.lat)))/(2.0**ZOOM_LEVEL) # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

        # trackpoint max accumulation per pixel = 1/5 (trackpoint/meter) * res_pixel (meter/pixel) * activities
        # (Strava records trackpoints every 5 meters in average for cycling activites)
        m = max(1.0, np.round((1.0/5.0)*res_pixel*len(activities.startTime.unique())))
    else:
        m = 1.0
    data[data > m] = m


    # equalize histogram and compute kernel density estimation
    if HEATMAP:
        data_hist, _ = np.histogram(data, bins=int(m+1))

        data_hist = np.cumsum(data_hist)/data.size # normalized cumulated histogram

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = m*data_hist[int(data[i, j])] # histogram equalization

        data = gaussian_filter(data, float(SIGMA)) # kernel density estimation with normal kernel

        data = (data-data.min())/(data.max()-data.min()) # normalize to [0,1]

    # colorize
    if HEATMAP:
        cmap = plt.get_cmap(PLT_COLORMAP)

        data_color = cmap(data)
        data_color[data_color == cmap(0.0)] = 0.0 # remove background color

        for c in range(3):
            frame[:, :, c] = (1.0-data_color[:, :, c])*frame[:, :, c]+data_color[:, :, c]

    else:
        if not SHOW_LEAD:
            color = np.array([255, 82, 0], dtype=float)/255 # orange
        

            for c in range(3):
                frame[:, :, c] = np.minimum(frame[:, :, c]+gaussian_filter(data, 1.0), 1.0) # white
                frame[:, :, c] = np.maximum(frame[:, :, c], 0.0)

            data = gaussian_filter(data, 0.5)
            data = (data-data.min())/(data.max()-data.min())

            

            for c in range(3):
                frame[:, :, c] = (1.0-data)*frame[:, :, c]+data*color[c]
        
        else:
            #color = mpl.cm.plasma()
            #color = np.array([255, 82, 0], dtype=float)/255 # orange

            #for c in range(3):
                #frame[:, :, c] = np.minimum(frame[:, :, c]+gaussian_filter(data, 1.0), 1.0) # white
            #    frame[:, :, c] = np.maximum(frame[:, :, c], 0.0)
            
            data = gaussian_filter(data, 0.5)
            
            data[data>1]=1
            data[data<0]=0
            #data = (data-data.min())/(data.max()-data.min())

            colorData=plt.get_cmap(PLT_COLORMAP)(data)

            dataBool=data>0.01

            for c in range(3):
                #frame[:,:,c]=(1-data)*fram[:,:colorData[:,:,c]*data
                frame[:, :, c] = (1.0-dataBool)*frame[:, :, c]+dataBool*colorData[:,:,c]


    # crop image
    

    frame = frame[max(i_min-MARGIN_SIZE, 0):min(i_max+MARGIN_SIZE, frame.shape[0]),
                            max(j_min-MARGIN_SIZE, 0):min(j_max+MARGIN_SIZE, frame.shape[1])]
    

    if SHOW_MILEAGE:
        def putCenterBottomAlignedText(text,x,y,fontScale,color,thickness,fontFace=cv2.FONT_HERSHEY_COMPLEX):
            global frame
            text_width, text_height = cv2.getTextSize(str(text), fontFace, fontScale, thickness)[0]

            frame=cv2.putText(frame, str(text), (int(x-text_width/2),int(y+text_height)), fontFace, 
                    fontScale, plt.get_cmap(PLT_COLORMAP)(255), thickness, cv2.LINE_AA)
        
        totalDist=df["distTotal"].max()
        if(math.isnan(totalDist)):
            totalDist=0
        putCenterBottomAlignedText(format(totalDist,'.1f')+"mi",frame.shape[0]/2*0.9,frame.shape[1],3,(0.5,0.5,0),5)

    plt.imsave(f"{FRAME_DIR}/img-{frameI}.jpg",frame)

#ffmpeg -framerate 30 -i frames/img-%d.jpg -c:v libx264 -r 30 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4