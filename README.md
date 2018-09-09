# eameo-faceswap-generator
Inspired by the amazing work done daily by [Eameo](https://twitter.com/EameoOk), I put together this simple script that
switch faces between two input images. You can run the step-by-step ipynb in [Google Colab](https://colab.research.google.com/github/nicolasmetallo/eameo-faceswap-generator/blob/master/face-swap-step-by-step.ipynb)
or run ```python faceswap.py``` directly.
# Installation
This script supports Python 2.7 and 3.6.
## Clone this repository
````
$ git clone https://github.com/nicolasmetallo/eameo-faceswap-generator.git
````
## Install Dlib
Run the following to install Dlib in Google Colaboratory.
````
!apt update
!apt install -y cmake
!pip install dlib
````
Click here for [MacOS/Ubuntu](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide) instructions or [Windows](https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only). 
## Pre-requisites
Run the following to install required modules
````
$ pip install -r requirements.txt
````
# Example Usage
This script requires the user to input three file paths which can be local paths or URLs:
- from_image = 'Path to first image where you will extract the face.'
- to_image = 'Path to second image where you will swap the existing face'
- output_filename = 'Path to output image.'

Now let's take two input images and set ```result.jpg``` as the output file.
```
$ python faceswap.py image1.jpg image2.jpg result.jpg
```
<img src="https://i.imgur.com/Wkya5C0.png">
