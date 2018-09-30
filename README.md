# Image Mosaic based on SIFT and RANSAC
**This is the project design of course *Digital Image Processing* (2017-2018, Fall) in Tsinghua University presented by Dr.Shengjin Wang**.

## Requirement
1. OpenCV 3.4.1 for python
2. OpenCV-Contrib 3.4.1
3. numpy 1.8.0rc1 or higher

## Intro
### SIFT
SIFT(Scale-invariant Feature Transform) is a computer vision algorithm presented by Dr.David Lowe in 1999. It can extract feature points of a image, and has been proved very efficient in Image Mosaic.

### RANSAC
RANSAC can be used to fit linear relation. Different from The Least Square Method, RANSAC is especially efficient for strong noisy data.

See `./Report.pdf` for more detail.

## Images Set
Put all the images you want to mosaic in `./Pics/`. The image files should be named in ascending order. The algorithm will mosaic them from left to right according to the index of file names.

For example:
```
ROOT/
└----Pics/
     |----table1.JPG
     |----table2.JPG
     |----table3.JPG
     |----table4.JPG
     └----table5.JPG
```

## Get Started!
Run:
```
python SIFT_Project.py
```
or:
```
python SIFT_Project.py --img_dir=YOUR_IMG_DIR --result_dir=YOUR_DIR --result_name=YOUR_NAME --format=YOUR_FORMAT
```
where:
```
--img_dir: The path of your images set. Default:'./Pics/'
--result_dir: Where the mosaic result should be saved. Default:'./Result/'
--result_name: The file name of the mosaic result. Default:'result.jpg'
--format: The file format of your images set. It can contain various format. Default:'jpg png'
```
The result image will be placed where you specify or in `./Result/` by default.
