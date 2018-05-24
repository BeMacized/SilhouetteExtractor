#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "nms.hpp"

// ERROR CODE(S)
const int NO_PERSON_DETECTED = 1;
const int NOT_VERTICAL = 2;
const int INVALID_SCALE = 3;

void debug(std::string msg) {
    std::cout << "[DEBUG] " << msg << std::endl;
}

void info(std::string msg) {
    std::cout << "[INFO] " << msg << std::endl;
}

void error(std::string msg) {
    std::cerr << "[ERR] " << msg << std::endl;
}

cv::Mat extractSilhouette(cv::Mat img, cv::Mat mask, double scale, int resolution) {
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    // Validate scale
    if (scale <= 0 || scale > 1) {
        throw INVALID_SCALE;
    }

    // Resize images for processing
    debug("Resizing input image");
    float resizeFactor = 1024.f / std::max(img.rows, img.cols);
    cv::resize(img, img, cv::Size(), resizeFactor, resizeFactor, cv::INTER_CUBIC);
    cv::resize(mask, mask, cv::Size(), resizeFactor, resizeFactor, cv::INTER_CUBIC);
    info("Resized input image");

    // Initialise person detector
    debug("Detecting people in source image");
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Detect people in the front image
    std::vector<cv::Rect> humanBounds;
    std::vector<double> weights;
    hog.detectMultiScale(img, humanBounds, weights, 0, cv::Size(4, 4), cv::Size(8, 8), 1.05);

    // Quit if none detected
    if (humanBounds.size() == 0) {
        error("No person detected");
        throw NO_PERSON_DETECTED;
    }
    info("Detected person in image");

    // Non-maxima suppression for merging detections
    debug("Applying non-maxima suppression to detected bounds");
    std::vector<std::vector<float>> procRects;
    for (cv::Rect bound : humanBounds) {
        procRects.push_back({bound.x, bound.y, bound.x + bound.width, bound.y + bound.height});
    }
    humanBounds = nms(procRects, 0.65);
    info("Applied non-maxima suppression to detected bounds");

    // Extract foreground
    {
        debug("Extracting foreground");
        // Threshold mask
        cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY);
        mask.setTo(cv::GC_BGD, mask == 0);
        mask.setTo(cv::GC_PR_FGD, mask == 255);

        // Execute grabcut
        cv::Mat bgModel, fgModel;
//        cv::Rect bounds(0,0,img.cols, img.rows);
        cv::grabCut(img, mask, humanBounds.front(), bgModel, fgModel, 10, cv::GC_INIT_WITH_MASK);
        // Write differences to mask
        cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);

        img = mask;
        info("Extracted foreground");
    }


    // Crop image to human bounds
    debug("Cropping image to detected bounds");
    img = cv::Mat(img, humanBounds.front());
    info("Cropped image to detected bounds");

    debug("Eroding & Dilating image");
    cv::erode(img, img, cv::Mat());
    cv::dilate(img, img, cv::Mat());
    info("Eroded & Dilated image");

    // Trim image
    {
        debug("Trimming image");
        cv::Mat boundingPoints;
        findNonZero(img, boundingPoints);
        img = cv::Mat(img, boundingRect(boundingPoints));
        if (img.cols == 0 && img.rows == 0) {
            img = cv::Mat(1, 1, img.type(), cv::Scalar(0, 0, 0));
        }
        info("Trimmed image");
    }

    if (img.cols > img.rows) {
        error("Silhouette not vertical");
        throw NOT_VERTICAL;
    }

    // Invert matte
    debug("Inverting image");
    cv::bitwise_not(img, img);
    info("Inverted image");

    // Draw matte to canvas at its relative scale
    {
        debug("Drawing silhouette at scale");
        int oRes = static_cast<int>(std::round(1. / scale * static_cast<double>(img.rows)));
        cv::Mat canvas(oRes, oRes, img.type(), cv::Scalar(255, 255, 255));
        int x = (oRes - img.cols) / 2;
        int y = oRes - img.rows;
        img.copyTo(canvas.colRange(x, x + img.cols).rowRange(y, y + img.rows));
        img = canvas;
        info("Drawn silhouette at scale");
    }

    // Resize canvas to resolution
    debug("Resizing canvas to required resolution");
    cv::resize(img, img, cv::Size(resolution, resolution), 1, 1, CV_INTER_LINEAR);
    info("Resized canvas to required resolution");

    return img;
}

int main(int argc, char **argv) {
    cv::Mat silhouette;
    try {
        silhouette = extractSilhouette(cv::imread("../images/fg1.jpg"), cv::imread("../images/mask1.jpg", CV_8UC1), .8, 512);
//        silhouette = extractSilhouette(cv::imread("../images/fg4.png"), cv::imread("../images/mask4.png", CV_8UC1), .8, 512);
    } catch (int errCode) {
        std::cerr << "An error occurred: ERR CODE " << errCode << std::endl;
        return errCode;
    }

    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    cv::imshow("Image", silhouette);
    cv::waitKey(0);
}

//int main(int argc, char **argv) {
// <BG_FP> <FG_FP> <OUT_FP> <HEIGHT> <MAX_HEIGHT> <RES> <THRESHOLD>

// Read front and back images
//    cv::Mat back = cv::imread(argv[1]);
//    cv::Mat front = cv::imread(argv[2]);

// Parse input values
//    int height = atoi(argv[4]);
//    int maxHeight = atoi(argv[5]);
//    int resolution = atoi(argv[6]);
//    int threshold = atoi(argv[7]);
//    if (threshold <= 0) threshold = 512;
//    double scale = static_cast<double>(height) / static_cast<double>(maxHeight);

// Extract silhouette
//    cv::Mat silhouette;
//    try {
//        silhouette = extractSilhouette(back, front, scale, resolution, threshold);
//    } catch (int errCode) {
//        std::cerr << "An error occurred: ERR CODE " << errCode << std::endl;
//        return errCode;
//    }

// Save image
//    cv::imwrite(argv[3], silhouette);
//
//    return 0;
//}