Computes the optical flow for a sequence of images, using the TV-L1 algorithm. The optical flow is calculated between every 3rd image.


### Build
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
### Run
    cd build
    ./calculate_optical_flow <input directory with RGB images> <output directory for flow images>

### Generate optical flow for dataset
For the place action dataset, you can use the script under `scripts` to generate both optical flow for the images from the camera and the rendered images.
Extract the train, test and validation sets and provide the path of the root folder to the script.

    cd scripts
    ./calc_optical_flow.sh <path to folder containing train, test and validation folders>
