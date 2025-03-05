#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>


cv::Mat averageFilter(const cv::Mat& image, int size = 3) {
    
    cv::Mat filteredImage = image.clone();
    
    int halfKernelSize = size / 2;
    
    cv::Mat avgKernel = cv::Mat::ones(size, size, CV_32F);
    
    avgKernel = avgKernel / (size * size);
    
    std::cout << "Average kernel: " << avgKernel << std::endl;
    
    // Check if image is grayscale or color
    if (image.channels() == 3) {
        for (int channel = 0; channel < 3; channel++) {
            for (int row = halfKernelSize; row < image.rows - halfKernelSize; row++) {
                for (int col = halfKernelSize; col < image.cols - halfKernelSize; col++) {
                    float sum = 0.0f;
                    
                    // Apply kerneld
                    for (int kr = -halfKernelSize; kr <= halfKernelSize; kr++) {
                        for (int kc = -halfKernelSize; kc <= halfKernelSize; kc++) {
                            sum += image.at<cv::Vec3b>(row + kr, col + kc)[channel] * 
                                   avgKernel.at<float>(kr + halfKernelSize, kc + halfKernelSize);
                        }
                    }
                    
                    // Update
                    filteredImage.at<cv::Vec3b>(row, col)[channel] = static_cast<uchar>(sum);
                }
            }
        }
    } else {
        // Grayscale image
        std::cout << "Grayscale image" << std::endl;
        for (int row = halfKernelSize; row < image.rows - halfKernelSize; row++) {
            for (int col = halfKernelSize; col < image.cols - halfKernelSize; col++) {
                float sum = 0.0f;
                
                // Apply kernel to neighborhood
                for (int kr = -halfKernelSize; kr <= halfKernelSize; kr++) {
                    for (int kc = -halfKernelSize; kc <= halfKernelSize; kc++) {
                        sum += image.at<uchar>(row + kr, col + kc) * 
                               avgKernel.at<float>(kr + halfKernelSize, kc + halfKernelSize);
                    }
                }
                
                // Update pixel value
                filteredImage.at<uchar>(row, col) = static_cast<uchar>(sum);
            }
        }
    }
    
    return filteredImage;
}


cv::Mat gaussianFilterCustom(const cv::Mat& image, int size = 3, double sigma = 1.0) {
    cv::Mat filteredImage = image.clone();
    
    int halfKernelSize = size / 2;
    
    cv::Mat gaussianKernel = createGaussianKernel(size, sigma);
    
    std::cout << "Gaussian kernel: " << gaussianKernel << std::endl;
    
    if (image.channels() == 3) {
        for (int channel = 0; channel < 3; channel++) {
            for (int row = halfKernelSize; row < image.rows - halfKernelSize; row++) {
                for (int col = halfKernelSize; col < image.cols - halfKernelSize; col++) {
                    float sum = 0.0f;
                    
                    // Apply kernel
                    for (int kr = -halfKernelSize; kr <= halfKernelSize; kr++) {
                        for (int kc = -halfKernelSize; kc <= halfKernelSize; kc++) {
                            sum += image.at<cv::Vec3b>(row + kr, col + kc)[channel] * 
                                   gaussianKernel.at<float>(kr + halfKernelSize, kc + halfKernelSize);
                        }
                    }
                    
                    // Update pixel value (clamp to 0-255)
                    filteredImage.at<cv::Vec3b>(row, col)[channel] = static_cast<uchar>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }
    } else {
        // Grayscale image
        for (int row = halfKernelSize; row < image.rows - halfKernelSize; row++) {
            for (int col = halfKernelSize; col < image.cols - halfKernelSize; col++) {
                float sum = 0.0f;
                
                for (int kr = -halfKernelSize; kr <= halfKernelSize; kr++) {
                    for (int kc = -halfKernelSize; kc <= halfKernelSize; kc++) {
                        sum += image.at<uchar>(row + kr, col + kc) * 
                               gaussianKernel.at<float>(kr + halfKernelSize, kc + halfKernelSize);
                    }
                }
                
                filteredImage.at<uchar>(row, col) = static_cast<uchar>(std::min(std::max(sum, 0.0f), 255.0f));
            }
        }
    }
    
    return filteredImage;
}


cv::Mat medianFilterCustom(const cv::Mat& image, int size = 3) {
    cv::Mat filteredImage = image.clone();
    
    int halfKernelSize = size / 2;
    
    std::vector<uchar> neighborhood;
    neighborhood.reserve(size * size);
    
    if (image.channels() == 3) {
        for (int channel = 0; channel < 3; channel++) {
            for (int row = halfKernelSize; row < image.rows - halfKernelSize; row++) {
                for (int col = halfKernelSize; col < image.cols - halfKernelSize; col++) {
                    // Clear neighborhood vector
                    neighborhood.clear();
                    
                    for (int kr = -halfKernelSize; kr <= halfKernelSize; kr++) {
                        for (int kc = -halfKernelSize; kc <= halfKernelSize; kc++) {
                            neighborhood.push_back(image.at<cv::Vec3b>(row + kr, col + kc)[channel]);
                        }
                    }
                    
                    // Sort neighborhood to find median
                    std::sort(neighborhood.begin(), neighborhood.end());
                    
                    uchar medianValue = neighborhood[neighborhood.size() / 2];
                    
                    filteredImage.at<cv::Vec3b>(row, col)[channel] = medianValue;
                }
            }
        }
    } else {
        for (int row = halfKernelSize; row < image.rows - halfKernelSize; row++) {
            for (int col = halfKernelSize; col < image.cols - halfKernelSize; col++) {
               neighborhood.clear();
                
                for (int kr = -halfKernelSize; kr <= halfKernelSize; kr++) {
                    for (int kc = -halfKernelSize; kc <= halfKernelSize; kc++) {
                        neighborhood.push_back(image.at<uchar>(row + kr, col + kc));
                    }
                }
                
                // Sort neighborhood to find median
                std::sort(neighborhood.begin(), neighborhood.end());
                
                uchar medianValue = neighborhood[neighborhood.size() / 2];
                
                // Update pixel value
                filteredImage.at<uchar>(row, col) = medianValue;
            }
        }
    }
    
    return filteredImage;
}


cv::Mat applyLowPassFilter(const cv::Mat& image, const std::string& filterType = "Average", 
                           int size = 3, double sigma = 1.0) {
    if (filterType == "Average") {
        return averageFilter(image, size);
    } else if (filterType == "Gaussian") {
        return gaussianFilterCustom(image, size, sigma);
    } else if (filterType == "Median") {
        return medianFilterCustom(image, size);
    } else {
        throw std::invalid_argument("Unknown filter type: " + filterType);
    }
}

cv::Mat createGaussianKernel(int size, double sigma) {
    int halfSize = size / 2;
    cv::Mat kernel(size, size, CV_32F);
    
    double sum = 0.0;
    
    // Create the kernel
    for (int y = -halfSize; y <= halfSize; y++) {
        for (int x = -halfSize; x <= halfSize; x++) {
            double value = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<float>(y + halfSize, x + halfSize) = static_cast<float>(value);
            sum += value;
        }
    }
    
    // Normalize the kernel
    kernel = kernel / sum;
    
    return kernel;
}























// Example of how to use these functions from Python with pybind11
/*
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Conversion functions between numpy arrays and cv::Mat would be needed here

PYBIND11_MODULE(cpp_filters, m) {
    m.doc() = "C++ implementation of image filtering algorithms";
    
    m.def("apply_low_pass_filter", &applyLowPassFilter, 
          "Apply low pass filter to an image",
          py::arg("image"), py::arg("filter_type") = "Average", 
          py::arg("size") = 3, py::arg("sigma") = 1.0);
}
*/
