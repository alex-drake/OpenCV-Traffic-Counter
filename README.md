# OpenCV-Traffic-Counter
This project details how to create a simple traffic counter designed using the OpenCV library for Python 3.5, and was originally carried out as part of the [Government Data Science Accelerator programme](https://gdsdata.blog.gov.uk/2017/08/11/pharmacies-people-and-ports-the-data-science-accelerator/) in June-October 2017.

<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/outputScreen.png?raw=true" width="352">

## The Project
The purpose of this project was to detect and count vehicles from a CCTV feed, and to ultimately give an idea of what the real-time on street situation is across the road network (in this case within Greater London). To that end, the TfL JamCam API was used throughout to test the algorithm. This is an API provided by Transport for London and can be used to obtain ~10 second clips of road traffic across the London road network, an example of which can be seen [here](https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00002.00625.mp4).

The main code can be found in [/trafficCounter/blobDetection.py](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/blobDetection.py) along with some other useful scripts that will assist with extracting [individual frames](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/frame_extract.py), [histograms](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/histogram_extraction.py) for illustrating how different conditions affect each frame, and [/trafficCounter/createSeedFiles.py](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/createSeedFiles.py) and [/trafficCounter/haarCascades.py](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/haarCascades.py) for starting work with HAAR cascades as an alternative method to blob detection (work in progress).

## Method
### Object Detection
In order to count vehicles we first need to be able to detect them in an image. This is pretty simple for a human to pick out but harder to implement in the machine world. However, if we consider that an image is just an array of numbers (one value per pixel), we may be able to use this to determine what a vehicle looks like and what we'd expect to see when there isn't a vehicle there. We can use OpenCV to look at how the value of certain pixels changes for these two conditions, as shown in the image below. To do this, we must first translate our image from RGB channels (Red, Green Blue) to HSV (Hue, Saturation, Value) and inspect each channel to see if it can tell us something.
<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/carNoCar.png?raw=true" width="400"><br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/hsv.png?raw=true" width="400"><br>
As we can see from the histogram plots, the Hue channel does not offer much information, whereas both the Saturation and Value channels clearly show a difference between the Vehicle/No Vehicle conditions and so we can use this channels in our detection algorithm. However, for simplicity we will just use the Value channel for the time being.

We can then use this information to determine what is background and what is a vehicle, so long as we have a suitable background image ie a version of our scene with no vehicles in it. In the case shown here it is very difficult to obtain a clear image, however we can use OpenCV to average between several frames and create our background image.
<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/backgrounds/625_bg.jpg?raw=true" width="352"><br>
Now that we have a background image, or an array of default/background values, we can use OpenCV to detect when these values go above a certain value (or 'threshold value'). We assume that this occurs when there is a vehicle within that pixel, and so use OpenCV to set the pixels that meet the threshold criteria to maximum brightness (this will make detecting shapes/vehicles easier later on).
<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/thresh.png?raw=true" width="352">
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/blobs.png?raw=true" width="352"><br>
The images above show the pixels that meet the threshold criteria (left) and the resulting shapes after setting those pixels to maximum value/brightness (right). Also highlighted (green) is gaps in our objects where dark areas (windscreens, grills etc) may not meet our threshold criteria. This could cause a problem later on so we try to fill in these gaps using the erosion and dilation functions from the OpenCV library.

Once we are happy with the shapes created, we must then check the shapes (or contours) to determine which are most like to be vehicles before dismissing those that are not. We can do this be implementing a condition where we are only interested in the detected contours if they are over a certain size. Note that this will change depending on the video feed. The kept contours can then be passed to the [Vehicle Counter algorithm](https://stackoverflow.com/a/36274515), based on the one created by Dan Maesk.

### Counting Vehicles
The vehicle counter is split into two class objects, one named `Vehicle` which is used to define each vehicle object, and the other `Vehicle Counter` which determines which 'vehicles' are valid before counting them (or not). `Vehicle` is relatively simple and offers information about each detected object such as a tracked position in each frame, how many frames it has appeared in (and how many it has not been seen for if we temporarily loose track of it), whether we have counted the vehicle yet and what direction we believe the vehicle to be travelling in. We can also obtain the last position and the position before that in order to calculate a few values within our `Vehicle Counter` algorithm.

`Vehicle Counter` is more complex and serves several purposes. We can use it to determine the vector movement of each tracked vehicle from frame to frame, giving an indicator of what movements are true and which are false matches. We do this to make sure we're not incorrectly matching vehicles and therefore getting the most accurate count possible. In this case, we only expect vehicles travelling from the top of the image to the bottom right hand corner, or the reverse. This means we only have a certain range of allowable vector movements based on the angle that the vehicle has moved - this can be seen from the images below. The image on the left shows the expected vector movements (highlighted in red) and the image on the left shows a chart of distance moved vs the angle - those classed as allowable movements are highlighted by the green box.
<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/vector.png?raw=true" width="175">
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/vectorMovements.png?raw=true" width="400"><br>
**Note that this section of code is ripe for improvements as the change in angle is likely a better indicator of a true match than absolute angle but this has not yet been implemented.**

If a vehicle object satisfies the above criteria, we then want to check what direction it is moving in before then passing it to the counter. We can then use this information to determine whether the vehicle should be counted and then whether the count applies to the left hand lanes (up direction) or right hand lanes (down direction). Once satisfied, we update the counter and print it to the output frame. If a vehicle has not been seen for a while, we remove it from the list of tracked objects as it is no longer of interest.

## Challenges and Improvements
The algorithm used works well in situations where traffic is free-flowing, within day-light hours. It also works relatively well in most weather conditions although background removal proves difficult in high winds as a moving camera means the background also changes quickly. However, accuracy drops when vehicles are either close together or have large shadows (forming one large object), dark vehicles do not always meet the detection criteria, and night scenes are difficult to resolve as headlight beams can create large areas that meet threshold criteria. Detection criteria are also relatively unique for each camera and so it may take time to refine these values to be confident in the output counts.

Many of these issues could be resolved by investigating alternative detection methods that do not rely so heavily on detecting pixels above a threshold value. To that end, detecting vehicles using HAAR cascades would potentially resolve these issues or at least provide a more accurate and consistent method for counting vehicles in various conditions and without worrying too much about initial detection values. That said, this would create the need for good training data and potentially data for each camera and so would add be more resource heavy initially.

## Resources / Useful Reading
* [Counting Cars Open CV (Dan Masek)](https://stackoverflow.com/a/36274515)
* [Speed Tracking (Ian Dees)](https://github.com/iandees/speedtrack)
* [SDC Vehicle Lane Detection (Max Ritter)](https://github.com/maxritter/SDC-Vehicle-Lane-Detection)
