#include <imreg_fmt/image_registration.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <set>
#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

void show_overlayed_image(const cv::Mat &prev_im, const cv::Mat &reg_im)
{
    cv::Mat overlay_image;
    cv::addWeighted(prev_im, 0.5, reg_im, 0.5, 0.0, overlay_image);
    cv::imshow("overlay_image", overlay_image);
    cv::waitKey(0);
}

int main(int argc, char **argv)
{
    std::string input_directory;
    if (argc < 2)
    {
        std::cout << "specify input image directory" << std::endl;
        return 1;
    }
    input_directory = std::string(argv[1]);

    if (!bfs::exists(input_directory) || !bfs::is_directory(input_directory))
    {
        std::cerr << "directory does not exist: " << input_directory << std::endl;
        return 1;
    }
    std::cout << input_directory << std::endl;

    bool debug = false;
    if (argc > 2)
    {
        debug = true;
    }


    std::set<std::string> sorted_images;
    bfs::directory_iterator end;
    for (bfs::directory_iterator iter(input_directory); iter != end; ++iter)
    {
        if (!bfs::is_directory(*iter))
        {
            sorted_images.insert(iter->path().string());
        }
    }

    cv::Mat im;
    cv::Mat registered_image;

    im = cv::imread(*(sorted_images.begin()), CV_LOAD_IMAGE_COLOR);

    cv::Rect roi_top;
    roi_top.x = 0;
    roi_top.y = 0;
    roi_top.width = im.cols;
    roi_top.height = (int)(0.4 * im.rows);

    cv::Rect roi_right;
    roi_right.x = (int)(0.6 * im.cols);
    roi_right.y = 0;
    roi_right.width = im.cols - roi_right.x;
    roi_right.height = im.rows;

    cv::Mat im_top = im(roi_top);
    cv::Mat im_right = im(roi_right);

    /*
     * We perform image registration on:
     * i) the full image
     * ii) a region of interest which covers the top of the image
     * iii) a region of interest which covers the right of the image
     */
    ImageRegistration image_registration_top(im_top);
    ImageRegistration image_registration_right(im_right);
    ImageRegistration image_registration(im);

    int frame_number = 0;
    cv::Mat output;
    int divisor = 1;

    // x, y, rotation, scale
    std::vector<double> transform_params_top(4, 0.0);
    std::vector<double> transform_params_right(4, 0.0);
    std::vector<double> transform_params(4, 0.0);

    std::set<std::string>::iterator iter = sorted_images.begin();

    while(iter != sorted_images.end())
    {
        ++iter;
        if (iter == sorted_images.end())
        {
            break;
        }
        frame_number++;
        if (frame_number % divisor !=0) continue;

        im = cv::imread(*iter, CV_LOAD_IMAGE_COLOR);
        if (im.empty())
        {
            std::cout << "done" << std::endl;
            break;
        }

        cv::Mat im_top = im(roi_top);
        cv::Mat im_right = im(roi_right);
        image_registration_top.registerImage(im_top, registered_image, transform_params_top, false);
        if (debug)
        {
            show_overlayed_image(image_registration_top.getCurrentImage(), registered_image);
        }
        image_registration_top.next();

        image_registration_right.registerImage(im_right, registered_image, transform_params_right, false);
        if (debug)
        {
            show_overlayed_image(image_registration_right.getCurrentImage(), registered_image);
        }
        image_registration_right.next();

        image_registration.registerImage(im, registered_image, transform_params, false);
        if (debug)
        {
            show_overlayed_image(image_registration.getCurrentImage(), registered_image);
        }
        image_registration.next();


        std::cout << transform_params_top[0] << ", " << transform_params_top[1]
                  << ", " << transform_params_top[2] << ", " << transform_params_top[3]
                  << ", " << transform_params_right[0] << ", " << transform_params_right[1]
                  << ", " << transform_params_right[2] << ", " << transform_params_right[3]
                  << ", " << transform_params[0] << ", " << transform_params[1]
                  << ", " << transform_params[2] << ", " << transform_params[3] << std::endl;
    }
    return 0;
}
