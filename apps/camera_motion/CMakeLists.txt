cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(image_motion)

add_subdirectory(imreg_fmt)

find_package(OpenCV REQUIRED)
find_package(Boost 1.46 COMPONENTS
    filesystem
    system
    REQUIRED
)

include_directories(
    imreg_fmt/include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_executable(compute_image_motion
    src/compute_image_motion.cpp
)

target_link_libraries(compute_image_motion
    imreg_fmt
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
