import glob
import gzip
import fitparse
from pathlib import Path
import shutil
import os
from fit2gpx import Converter
import csv
import pandas as pd
import datetime as dt
import re
from tqdm import tqdm
import warnings




ACTIVITY_FILES="export/activities" #Location of the .gpx/.fit/.gz etc files
PROCESSED_FILES="processed_files" #Where to output after processing
ACTIVITY_METADATA="export/activities.csv" #Location of the file metadata

#Customize this function to return true or false if we want to process the activity based on its metadata
#The info argument is given the full row of data in activities.csv
def activityFilter(info):
    activityType=info["Activity Type"].values[0]
    dateStr=info["Activity Date"].values[0]

    datetime = dt.datetime.strptime(dateStr, "%b %d, %Y, %I:%M:%S %p")

    if activityType != "Run":
        return False
    if datetime.year<2022:
        return False
    
    return True

#Delete our old processed files folder
shutil.rmtree(PROCESSED_FILES,ignore_errors=True)

print("Reading metadata...")

metadata=pd.read_csv(ACTIVITY_METADATA)
print("Metadata has ",metadata.shape[1]," fields for",metadata.shape[0],"activities")

files=glob.glob(ACTIVITY_FILES+"/*")
print("found",len(files),"activity files")
 
#create output directory
Path(PROCESSED_FILES).mkdir(parents=True, exist_ok=True)


noMetaCount=0
failedFilterCount=0
unzippedCount=0
copiedCount=0

files=files
for path in files:
    filename=path[path.index("/")+1:]#this is the format that the filename will appear in activities.csv
    
    #get metadata
    info=metadata.loc[metadata["Filename"]==filename]

    #check if metadata is missing
    if(len(info)==0):
        noMetaCount+=1
        continue

    #check if we should filter out the activity or not
    if(not activityFilter(info)):
       failedFilterCount+=1
       continue

    
    
    #If it is a .gz we unzip to the output folder
    if(path[-3:]==".gz"):
        outputFile=PROCESSED_FILES+"/"+filename.split("/")[1][:-3]#Same file name but no .gz
        with gzip.open(path, 'rb') as f_in:
            with open(outputFile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        unzippedCount+=1
    #If it's not a gz we copy to the output folder
    else:
        shutil.copy(path, PROCESSED_FILES+"/"+filename.split("/")[1]) 
        copiedCount+=1


print(len(files),"files found.",noMetaCount," files had no metadata.",failedFilterCount," were filtered out.",unzippedCount," were unzipped.",copiedCount,"were copied directly.")

#convert fit to gpx
print("Converting fit to gpx...")

errorFiles=[]

conv = Converter()
toConvert=glob.glob(PROCESSED_FILES+"/*.fit")
for file in tqdm(toConvert):  
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:  
            gpx = conv.fit_to_gpx(f_in=file, f_out=file[:-4]+".gpx")
        except Exception:
            errorFiles.append(file)
            print("error: "+file)
        #print(file)
        Path(file).unlink()#delete converted file


print(len(toConvert)-len(errorFiles),"converted successfully.",len(errorFiles),"files could not be converted from fit to gpx")
