# Eye of Horus

The "Eye of Horus" is a concept and symbol in ancient Egyptian religion that represents well-being, healing, and protection. The Eye of Horus symbol, a stylized eye with distinctive markings, was believed to have protective magical power and appeared frequently in ancient Egyptian art.

<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/svisions/blob/main/wiki/wss-banner-staff-gage.jpg">
  Figure 1. A staff gage gives a quick estimate of gage height (stage) of a river. Photo by <a href="https://www.usgs.gov/special-topics/water-science-schoo">Water Science School</a> on <a href="https://www.usgs.gov/media/images/a-staff-gage-gives-a-quick-estimate-gage-height-stage-a-river">USGS.gov</a>.
</p>

## Table of contents

- [Field Survey](#Field-Survey)
- [Raspberry Pi](#Raspberry-Pi)
- [Computer Vision](#Computer-Vision)
- [Machine Learning](#Machine-Learning)


## Field Survey
## Raspberry Pi
## Computer Vision
### Camera Calibration
<!-- https://learnopencv.com/camera-calibration-using-opencv/ -->
The process of estimating the parameters of a camera is called camera calibration.

This means we have all the information (parameters or coefficients) about the camera required to determine an accurate relationship between a 3D point in the real world and its corresponding 2D projection (pixel) in the image captured by that calibrated camera.

Typically this means recovering two kinds of parameters. First, internal parameters of the camera/lens system e.g., focal length, optical center, and radial distortion coefficients of the lens. Second, external parameters refering to the orientation (rotation and translation) of the camera with respect to some world coordinate system.

#### Understanding Lens Distortion
<!-- https://learnopencv.com/understanding-lens-distortion/ -->
<!-- https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html -->
To generate clear and sharp images the diameter of the aperture (hole) of a pinhole camera should be as small as possible. If we increase the size of the aperture, we know that rays from multiple points of the object would be incident on the same part of the screen creating a blurred image. On the other hand, if we make the aperture size small, only a small number of photons hit the image sensor. As a result the image is dark and noisy.
  
So, smaller the aperture of the pinhole camera, more focused is the image but, at the same time, darker and noisier it is. While, with a larger aperture, the image sensor receives more photons (and hence more signal). This leads to a bright image with only a small amount of noise.
  
How do we get a sharp image but at the same time capture more light rays to make the image bright? We replace the pinhole by a lens thus increasing the size of the aperture through which light rays can pass. A lens allows larger number of rays to pass through the hole and because of its optical properties it can also focus them on the screen. This makes the image brighter. So we have bright and sharp, focused image using a lens.
By using a lens we get better quality images but the lens introduces some distortion effects. There are two major types of distortion effects :
  
* **Radial distortion**: This type of distortion usually occur due unequal bending of light. The rays bend more near the edges of the lens than the rays near the centre of the lens. Due to radial distortion straight lines in real world appear to be curved in the image. The light ray gets displaced radially inward or outward from its ideal location before hitting the image sensor. There are two type of radial distortion effect

  * Barrel distortion effect, which corresponds to negative radial displacement

  * Pincushion distortion effect, which corresponds to a positive radial displacement.

* **Tangential distortion**: This usually occurs when image screen or sensor is at an angle with respect to the lens, i.e., lenses are not perfectly parallel to the imaging plane. Thus the image seem to be tilted and stretched.

Radial distortion can be represented as follows:

x<sub>u</sub> = x(1 + k<sub>1</sub>r<sup>2</sup> + k<sub>2</sub>r<sup>4</sup> + k<sub>3</sub>r<sup>6</sup>)

y<sub>u</sub> = y(1 + k<sub>1</sub>r<sup>2</sup> + k<sub>2</sub>r<sup>4</sup> + k<sub>3</sub>r<sup>6</sup>)

Where, (x<sub>u</sub>, y<sub>u</sub>) represents pixel point coordinates on the corrected output image.

The amount of tangential distortion can be represented as below:

x<sub>u</sub> = x + [2p<sub>1</sub>xy + p<sub>2</sub>(r<sup>2</sup> + 2x<sup>2</sup>)]

y<sub>u</sub> = y + [p<sub>1</sub>(r<sup>2</sup> + 2y<sup>2</sup>) + 2p<sub>2</sub>xy]


we need to find five parameters, known as distortion coefficients given by:

Distortion Coefficients: (k<sub>1</sub>, k<sub>2</sub>, p<sub>1</sub>, p<sub>2</sub>, k<sub>3</sub>)

#### Intrinsic Parameters
Intrinsic parameters are specific to a camera. They include information like focal length (f<sub>x</sub>, f<sub>y</sub>) and optical centers (c<sub>x</sub>, c<sub>y</sub>). The focal length and optical centers can be used to create a camera matrix, which can be used to remove distortion due to the lenses of a specific camera. The camera matrix is unique to a specific camera, so once calculated, it can be reused on other images taken by the same camera. It is expressed as a 3x3 matrix.

#### Extrinsic parameters
Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system.

### Spatial Resection
<!--   https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/ -->
### 3D to 2D Projection
<!--   https://learnopencv.com/geometry-of-image-formation/ -->
## Machine Learning

<!-- 1- Download the chessboard pattern [here](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png). -->
<!-- ## STEP 2: Spatial Resection
## STEP 3: 3D Point Cloud Transformation
## STEP 3: 3D to 2D Reprojection -->


