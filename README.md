# Strava animated heatmap generator

<details>
<summary><a>Example 1</a></summary>
<img src="readme_files/example.gif" alt="drawing" width="300"/>
</details>


<details>
<summary><a>Example 2</a></summary>
<img src="readme_files/example2.gif" alt="drawing" width="300"/>
</details>

## How to

### Step 1: Downloads
A. Clone or download this repository. 

B. `cd` into the directory and create and activate the virtual enviroment: `python3 -m venv env` followed by `source env/bin/activate` 

B. Install the requirements with `pip install -r requirements.txt`

C. Request and download data archive from Strava. Unzip and copy the folder into the same directory as this readme.
<a href="https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export#:~:text=Choose%20%22Settings%2C%22%20then%20find,may%20take%20a%20few%20hours.)">Instructions here</a>

### Step 2: Process strava files

The file `process_files.py` has 3 functions: 
 - Unzips any `.gz` files in the list of exported activities
 - Converts `.fit` files to `.gpx`
 - Reads the activity metadata file and filters the `gpx` files based on their metadata.

First, open up `process_files.py` and edit these three lines according to the data you copied in and your preferred output directory.

```python
ACTIVITY_FILES="export/activities"
PROCESSED_FILES="processed_files"
ACTIVITY_METADATA="export/activities.csv"
```

Next, edit this function to choose which activities to include in the output. 
```python
def activityFilter(info):
    ...
```
You can do additional filtering later, but this function lets you do basic like selecting a date range or a specific activity type in order to reduce the amount of data in later steps. The full row of metadata available for filtering can be found in the `activities.csv` file 

Finally, run `python process_files.py`.

### 3. Generate Video frames
This function will generate a dataframe of trackpoints from the processed gpx files, download map tiles from openstreetmap.org and generate a series of frames to be converted into video.

To speed up processing the openstreetmap frames and trackpoint dataframe are saved locally. To regenerate this data, you can delet just the files (`tiles/*` or `extractedData.pickle`) and re-run the main file.

Edit the activity filters in these lines:
```python
MIN_START_LAT=41.31
MAX_START_LAT=41.32
MIN_START_LON=-72.93
MAX_START_LON=-72.92
MIN_DATE=dt.datetime(2022,6,1).date()
```
You can do more filtering by editing 
```def processActivity(gpx):```

These constants specify some different visual effects for the animation
```python
ZOOM_LEVEL=13

DARK_MAP=True
SIGMA=2#size of the trail
HEATMAP=False
MARGIN_SIZE=32#in pixels
PLT_COLORMAP = 'hot' # matplotlib color map
SHOW_LEAD=True
```

`ZOOM LEVEL` is an important constant to play with in order to make sure that the output images are of the correct size. (This will depend on the size of the physical area over which your activities occur.)

One important parameter is the line defining `activities["showPoint"]`. This column should be a float between 0 and 1 representing when a specific gpx point should become visible in the animation. e.g. a `showPoint=0.5` will cause that specific trackpoint to become visible halfway through the animation
 

### 4. Turn video frames into animation

You will need to install ffmpeg for this step.

The command that worked for me is:  
`ffmpeg -framerate 30 -i frames/img-%d.jpg -c:v libx264 -r 30 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4`


