#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "nms.hpp"

// ERROR CODE(S)
const int NO_PERSON_DETECTED = 1;
const int NOT_VERTICAL = 2;
const int INVALID_SCALE = 3;

cv::Mat extractSilhouette(cv::Mat back, cv::Mat front, double scale, int resolution, double threshold = 512) {

    // Validate scale
    if (scale <= 0 || scale > 1) {
        throw INVALID_SCALE;
    }

    // Resize images for processing
    float resizeFactor = 1024.f / std::max(front.rows, front.cols);
    cv::resize(front, front, cv::Size(), resizeFactor, resizeFactor, cv::INTER_CUBIC);
    cv::resize(back, back, cv::Size(), resizeFactor, resizeFactor, cv::INTER_CUBIC);

    // Initialise person detector
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Detect people in the front image
    std::vector<cv::Rect> rects;
    std::vector<double> weights;
    hog.detectMultiScale(front, rects, weights, 0, cv::Size(4, 4), cv::Size(32, 32), 1.05);

    // Quit if none detected
    if (rects.size() == 0) throw NO_PERSON_DETECTED;

    // Non-maxima suppression for merging detections
    std::vector<std::vector<float>> procRects;
    for (cv::Rect rect : rects) {
        procRects.push_back({rect.x, rect.y, rect.x + rect.width, rect.y + rect.height});
    }
    rects = nms(procRects, 0.65);

    // Crop images to first detection
    front = cv::Mat(front, rects.front());
    back = cv::Mat(back, rects.front());

    // Initialize cv::Matte, BG subtractor & Contour list
    cv::Mat matte;
    std::vector<std::vector<cv::Point>> contours;
    cv::Ptr<cv::BackgroundSubtractor> bg = cv::createBackgroundSubtractorMOG2(0, threshold, false);

    // Subtract BG and write diff to matte
    bg->apply(back, matte);
    bg->apply(front, matte);
    erode(matte, matte, cv::Mat());
    dilate(matte, matte, cv::Mat());

    // Find contours and draw onto matte
    findContours(matte.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::drawContours(matte, contours, -1, cv::Scalar(255, 255, 255), 1);

    // Trim matte
    cv::Mat boundingPoints;
    findNonZero(matte, boundingPoints);
    matte = cv::Mat(matte, boundingRect(boundingPoints));
    if (matte.cols == 0 && matte.rows == 0) {
        matte = cv::Mat(1, 1, matte.type(), cv::Scalar(0,0,0));
    }

    // Validate orientation
    if (matte.cols > matte.rows) throw NOT_VERTICAL;

    // Invert matte
    bitwise_not(matte, matte);

    // Draw matte to canvas at its relative scale
    int oRes = static_cast<int>(std::round(1. / scale * static_cast<double>(matte.rows)));
    cv::Mat canvas(oRes, oRes, matte.type(), cv::Scalar(255, 255, 255));
    int x = (oRes - matte.cols) / 2;
    int y = oRes - matte.rows;
    matte.copyTo(canvas.colRange(x, x + matte.cols).rowRange(y, y + matte.rows));

    // Resize canvas to resolution
    cv::resize(canvas, canvas, cv::Size(resolution, resolution), 1, 1, CV_INTER_LINEAR);

    return canvas;
}

int main(int argc, char **argv) {
    // <BG_FP> <FG_FP> <OUT_FP> <HEIGHT> <MAX_HEIGHT> <RES> <THRESHOLD>

    // Read front and back images
    cv::Mat back = cv::imread(argv[1]);
    cv::Mat front = cv::imread(argv[2]);

    // Parse input values
    int height = atoi(argv[4]);
    int maxHeight = atoi(argv[5]);
    int resolution = atoi(argv[6]);
    int threshold = atoi(argv[7]);
    if (threshold <= 0) threshold = 512;
    double scale = static_cast<double>(height) / static_cast<double>(maxHeight);

    // Extract silhouette
    cv::Mat silhouette;
    try {
        silhouette = extractSilhouette(back, front, scale, resolution, threshold);
    } catch (int errCode) {
        std::cerr << "An error occurred: ERR CODE " << errCode << std::endl;
        return errCode;
    }

    // Save image
    cv::imwrite(argv[3], silhouette);

    return 0;
}