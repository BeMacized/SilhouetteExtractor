#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "nms.hpp"

// ERROR CODE(S)
const int NO_PERSON_DETECTED = 100;
const int NOT_VERTICAL = 101;
const int INVALID_SCALE = 102;

void logdebug(std::string msg) {
    std::cout << "[DEBUG] " << msg << std::endl;
}

void loginfo(std::string msg) {
    std::cout << "[INFO] " << msg << std::endl;
}

void logerror(std::string msg) {
    std::cerr << "[ERR] " << msg << std::endl;
}

cv::Mat getMask(int sourceCols, int sourceRows, std::string maskPath, double areaFactor) {
    // Load mask
    cv::Mat mask = cv::imread(maskPath, CV_8UC4);
    std::vector<cv::Mat> channels;
    cv::split(mask, channels);
    mask = channels[3];
    cv::threshold(mask, mask, 0, 255, CV_THRESH_BINARY);
    // Calculate max boundaries
    double maxWidth = static_cast<double>(sourceCols) * areaFactor;
    double maxHeight = static_cast<double>(sourceRows) * areaFactor;
    // Reference mask dimensions as doubles for later use
    double maskWidth = static_cast<double>(mask.cols);
    double maskHeight = static_cast<double>(mask.rows);
    // Determine scale factor for mask to fit in max boundaries
    double scale = std::fmin(maxWidth / maskWidth, maxHeight / maskHeight);
    // Calculate resulting width and height
    int width = static_cast<int>(std::floor(std::fmin(std::round(scale * static_cast<double>(mask.cols)), maxWidth)));
    int height = static_cast<int>(std::floor(std::fmin(std::round(scale * static_cast<double>(mask.rows)), maxHeight)));
    // Calculate offset position of mask to canvas
    int x = static_cast<int>(std::round(static_cast<double>(sourceCols - width) / 2.));
    int y = static_cast<int>(std::round(static_cast<double>(sourceRows - height) / 2.));
    // Resize mask
    cv::resize(mask, mask, cv::Size(width, height), 1, 1, CV_INTER_LINEAR);
    // Position mask
    {
        cv::Mat canvas(sourceRows, sourceCols, CV_8UC1, cv::Scalar(0));
        mask.copyTo(canvas.rowRange(y, y + height).colRange(x, x + width));
        mask = canvas;
    }
    // Produce dilated mask
    cv::Mat maskPRBG;
    cv::dilate(mask, maskPRBG, cv::Mat(), cv::Point(-1, -1), 20);
    // Produce eroded mask
    cv::Mat maskFG;
    cv::erode(mask, maskFG, cv::Mat(), cv::Point(-1, -1), 20);
    // Create canvas with GC_BGD
    cv::Mat canvas(sourceRows, sourceCols, CV_8UC1, cv::Scalar(cv::GC_BGD));
    // Write GC_PR_BGD to mask
    canvas.setTo(cv::GC_PR_BGD, maskPRBG == 255);
    // Write GC_PR_FGD to canvas
    canvas.setTo(cv::GC_PR_FGD, mask == 255);
    // Write GC_FGD to canvas
    canvas.setTo(cv::GC_FGD, maskFG == 255);
    // Return canvas
    return canvas;
}

cv::Mat extractSilhouette(cv::Mat img, std::string maskPath, double maskAreaFactor, double scale, int resolution) {
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    // Validate scale
    if (scale <= 0 || scale > 1) {
        throw INVALID_SCALE;
    }

    // Resize images for processing
    logdebug("Resizing input image");
    float resizeFactor = 1024.f / std::max(img.rows, img.cols);
    cv::resize(img, img, cv::Size(), resizeFactor, resizeFactor, cv::INTER_CUBIC);
    loginfo("Resized input image");

    // Initialise person detector
    logdebug("Detecting people in source image");
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Detect people in the front image
    std::vector<cv::Rect> humanBounds;
    std::vector<double> weights;
    hog.detectMultiScale(img, humanBounds, weights, 0, cv::Size(4, 4), cv::Size(8, 8), 1.05);

    // Quit if none detected
    if (humanBounds.size() == 0) {
        logerror("No person detected");
        throw NO_PERSON_DETECTED;
    }
    loginfo("Detected person in image");

    // Non-maxima suppression for merging detections
    logdebug("Applying non-maxima suppression to detected bounds");
    std::vector<std::vector<float>> procRects;
    for (cv::Rect bound : humanBounds) {
        procRects.push_back({bound.x, bound.y, bound.x + bound.width, bound.y + bound.height});
    }
    humanBounds = nms(procRects, 0.65);
    loginfo("Applied non-maxima suppression to detected bounds");

    // Extract foreground
    {
        logdebug("Extracting foreground");
        // Obtain mask
        cv::Mat mask = getMask(img.cols, img.rows, maskPath, maskAreaFactor);
        // Execute grabcut
        cv::Mat bgModel, fgModel;
        cv::grabCut(img, mask, humanBounds.front(), bgModel, fgModel, 10, cv::GC_INIT_WITH_MASK);
        // Transform mask to silhouette on img
        img = cv::Mat(mask.rows, mask.cols, CV_8UC1, cv::Scalar(0));
        img.setTo(255, mask == cv::GC_PR_FGD);
        img.setTo(255, mask == cv::GC_FGD);
        loginfo("Extracted foreground");
    }

    // Crop image to human bounds
    logdebug("Cropping image to detected bounds");
    img = cv::Mat(img, humanBounds.front());
    loginfo("Cropped image to detected bounds");

    logdebug("Eroding & Dilating image");
    cv::erode(img, img, cv::Mat());
    cv::dilate(img, img, cv::Mat());
    loginfo("Eroded & Dilated image");

    // Trim image
    {
        logdebug("Trimming image");
        cv::Mat boundingPoints;
        findNonZero(img, boundingPoints);
        img = cv::Mat(img, boundingRect(boundingPoints));
        if (img.cols == 0 && img.rows == 0) {
            img = cv::Mat(1, 1, img.type(), cv::Scalar(0, 0, 0));
        }
        loginfo("Trimmed image");
    }

    if (img.cols > img.rows) {
        logerror("Silhouette not vertical");
        throw NOT_VERTICAL;
    }

    // Invert matte
    logdebug("Inverting image");
    cv::bitwise_not(img, img);
    loginfo("Inverted image");

    // Draw matte to canvas at its relative scale
    {
        logdebug("Drawing silhouette at scale");
        int oRes = static_cast<int>(std::round(1. / scale * static_cast<double>(img.rows)));
        cv::Mat canvas(oRes, oRes, img.type(), cv::Scalar(255, 255, 255));
        int x = (oRes - img.cols) / 2;
        int y = oRes - img.rows;
        img.copyTo(canvas.colRange(x, x + img.cols).rowRange(y, y + img.rows));
        img = canvas;
        loginfo("Drawn silhouette at scale");
    }

    // Resize canvas to resolution
    logdebug("Resizing canvas to required resolution");
    cv::resize(img, img, cv::Size(resolution, resolution), 1, 1, CV_INTER_LINEAR);
    loginfo("Resized canvas to required resolution");

    return img;
}

int main(int argc, char **argv) {
    // <IMG_FP> <MASK_FP> <OUT_FP> <MASK_AREA_FACTOR> <HEIGHT> <MAX_HEIGHT> <RES>

    // Read image
    cv::Mat img = cv::imread(argv[1]);

    // Parse input values
    std::string maskPath = argv[2];
    double maskAreaFactor = atof(argv[4]);
    int height = atoi(argv[5]);
    int maxHeight = atoi(argv[6]);
    int resolution = atoi(argv[7]);
    double scale = static_cast<double>(height) / static_cast<double>(maxHeight);

    // Extract silhouette
    cv::Mat silhouette;
    try {
        silhouette = extractSilhouette(img, maskPath, maskAreaFactor, scale, resolution);
    } catch (int errCode) {
        std::cerr << "An error occurred: ERR CODE " << errCode << std::endl;
        return errCode;
    }

    // Save image
    cv::imwrite(argv[3], silhouette);

    return 0;
}