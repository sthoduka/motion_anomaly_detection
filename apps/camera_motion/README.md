Compute expected and observed pixel motion caused by camera motion.

## Expected pixel motion
The expected pixel motion caused by known/measured camera motion can be computed using epipolar geometry.
The camera motion is measured with the robot's proprioceptive sensors from the head, torso etc. Ultimately,
this is the transformation between the base frame and camera frame.
For computing the expected pixel motion, the intrinsic parameters of the camera are also needed. The
parameters used here are from the camera used in generating our dataset.

### Run
The script will compute the expected pixel motion (in `x` and `y`) for every third time step and save it in a Numpy file in the trial folder.

    cd scripts
    python compute_expected_pixel_motion.py <path to trial folder>

## Observed pixel motion
The observed pixel motion is computed as a similarity transform (translation, rotation, scale) by registering images using the [Fourier-Mellin transform](https://github.com/sthoduka/imreg_fmt). Here, we perform image registration on the full image, and on two patches of the image (patches at the top and right of the image). This is to account for cases in which the camera motion is not the dominant motion in the image. We then compare the expected pixel motion with all three similarity transforms, and choose the one with the minimum error.

### Compile
The [imreg_fmt](https://github.com/sthoduka/imreg_fmt) project is included as a submodule. If submodules have not been initialized yet, run the following:

    git submodule update --init

In order to compile the code, run:

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

### Run
The similarity transforms are output to stdout as comma separated values in the order x, y, rotation and scale for the top, right and full image.

    ./compute_image_motion <path to rgb images> > <path to trial folder/observed_pixel_motion.txt>

For e.g.:

    ./compute_image_motion /media/ubuntu/data/validation/trial_115/rgb > /media/ubuntu/data/validation/trial_115/observed_pixel_motion.txt

