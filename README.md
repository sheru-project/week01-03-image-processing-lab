# Weeks 1-3 Image Processing Lab

## Setup Instructions
To set up the environment and install required packages, run:
```bash
pip install -r requirements.txt

```

## Execution Command

To execute the script and generate all required output images, run from the root repository directory:

```bash
python3 weeks01-03_image_lab.py

```

## Task 1: Image Data Report

* **Width**: 1600 pixels
* **Height**: 1569 pixels
* **Channels**: 3
* **NumPy Shape**: `[1569, 1600, 3]`
* **Pixel Count**: 2,510,400 pixels
* **Estimated Data Size**: 7,531,200 bytes (~7.53 MB at 8 bits/channel)
* **Color Order**: `BGR`
* **Explanation**: OpenCV loads images as a 3D NumPy array organized by `[Height, Width, Channels]`. Height corresponds to rows (1569), Width corresponds to columns (1600), and the 3 channels represent color intensity in BGR (Blue, Green, Red) order.

## Task 2: Color and Resolution Observations

* **Original Dimensions**: 1600 x 1569 pixels
* **Downsampled Dimensions**: 800 x 784 pixels (half width and height)
* **Channel Difference**: The orange/yellow pen body appears significantly darker/black in the Blue channel compared to the Red and Green channels because orange/yellow reflects high amounts of red and green light while absorbing blue light.
* **Lost Detail**: Fine handwriting stroke edges and minor paper/shadow textures visible in the original RGB image become pixelated and smoothed in the downsampled image.

## Task 3: Adjustments Documentation

* **Brightness (Delta = +40)**: Shifts all pixel intensities up by 40, making the background paper whiter while clipping brighter areas to 255.
* **Contrast (Factor = 1.5)**: Multiplies intensity values by 1.5, widening the gap between light background and dark pen/text strokes, making text stand out dramatically.
* **Threshold (Val = 127)**: Converts grayscale to binary black-and-white. Pixels darker than 127 (the pen, text, and heavy laptop edge) become pure black (0), while paper background becomes pure white (255).

## Task 4: Blur and Edge Observations

* **Visible Edge Comparison**: In the original Sobel output, fine noise from paper texture and background grain creates scattered white specks across the black background.
* **Detail Reduced by Blur**: Applying a 5x5 mean blur smooths out high-frequency surface noise, resulting in a cleaner edge map in "Sobel Edges (Blurred)" where only major structural contours (text writing, pen boundary, laptop edge) remain clearly defined.

## References & External Sources

* OpenCV Documentation (`cv2.imread`, `cv2.Sobel`, `cv2.threshold`, `cv2.blur`, `cv2.resize`)
* NumPy array manipulation and Matplotlib subplots documentation
