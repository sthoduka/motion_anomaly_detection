Compute expected and observed pixel motion caused by camera motion.

## Expected pixel motion
The expected pixel motion caused by known/measured camera motion can be compute using epipolar geometry.
The camera motion is measured with the robot's proprioceptive sensors from the head, torso etc. Ultimately,
this is the transformation between the base frame and camera frame.
For computing the expected pixel motion, the intrinsic parameters of the camera are also needed. The
parameters used here are from the camera used in generating our dataset.

### Run
The script will compute the expected pixel motion (in `x` and `y`) for every third time step and save it in a Numpy file in the trial folder.

    cd scripts
    python compute_expected_pixel_motion.py <path to trial folder>

## Observed pixel motion
The observed pixel motion is computed as a similarity transform (translation, rotation, scale) by registering images using the [Fourier-Mellin transform](https://github.com/sthoduka/imreg_fmt).

### Compile
The [imreg_fmt](https://github.com/sthoduka/imreg_fmt) project is included as a submodule. If submodules have not been initialized yet, run the following:

    git submodule update --init

In order to compile the code, run:

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
