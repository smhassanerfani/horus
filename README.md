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

* **Tangential distortion**: This usually occurs when image screen or sensor is at an angle w.r.t the lens. Thus the image seem to be tilted and stretched.

Radial distortion can be represented as follows:

The amount of tangential distortion can be represented as below:

we need to find five parameters, known as distortion coefficients given by:

  * Intrinsic Parameters
  * Extrinsic parameters

### Spatial Resection
<!--   https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/ -->
### 3D to 2D Projection
<!--   https://learnopencv.com/geometry-of-image-formation/ -->
## Machine Learning

<!-- 1- Download the chessboard pattern [here](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png). -->
<!-- ## STEP 2: Spatial Resection
## STEP 3: 3D Point Cloud Transformation
## STEP 3: 3D to 2D Reprojection -->


