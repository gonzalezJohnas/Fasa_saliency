#ifndef _FASA_H_
#define _FASA_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>


class Fasa{

    public :

    Fasa();

    //////////////////////// INPUT PARAMETERS START ////////////////////////

        float sigmac;   // The sigma values that are used in computing the color weights

    // (WARNING!! Division operations are implemented with bitshift operations for efficiency. Please set "histogramSize1D" to a power of two)

        int histogramSize1D;    // Number of histogram bins per channel



    //////////////////////// INPUT PARAMETERS END ////////////////////////

        int histogramSize2D = histogramSize1D * histogramSize1D;
        int histogramSize3D = histogramSize2D * histogramSize1D;
        int logSize = (int) log2(histogramSize1D);
        int logSize2 = 2 * logSize;

        cv::Mat squares;

        float *squaresPtr;

        std::vector<cv::Mat> LAB;

        std::vector<float> L, A, B;

        float meanVectorFloat[4] = {0.5555, 0.6449, 0.0002, 0.0063};

        float inverseCovarianceFloat[4][4] = {{43.3777, 1.7633,  -0.4059, 1.0997},
                                              {1.7633,  40.7221, -0.0165, 0.0447},
                                              {-0.4059, -0.0165, 87.0455, -3.2744},
                                              {1.0997,  0.0447,  -3.2744, 125.1503}};

        cv::Mat modelMean ;
        cv::Mat modelInverseCovariance;


        void computeSaliencyMap(cv::Mat shapeProbability,
                                cv::Mat contrast,
                                cv::Mat exponentialColorDistance,
                                cv::Mat histogramIndex,
                                int *mapPtr,
                                cv::Mat &SM,
                                cv::Mat &saliency);

        void calculateProbability(cv::Mat mx,
                                  cv::Mat my,
                                  cv::Mat Vx,
                                  cv::Mat Vy,
                                  cv::Mat modelMean,
                                  cv::Mat modelInverseCovariance,
                                  int width,
                                  int height,
                                  cv::Mat &Xsize,
                                  cv::Mat &Ysize,
                                  cv::Mat &Xcenter,
                                  cv::Mat &Ycenter,
                                  cv::Mat &shapeProbability);

        void bilateralFiltering(cv::Mat colorDistance,
                                cv::Mat exponentialColorDistance,
                                std::vector<int> reverseMap,
                                int *histogramPtr,
                                float *averageXPtr,
                                float *averageYPtr,
                                float *averageX2Ptr,
                                float *averageY2Ptr,
                                cv::Mat &mx,
                                cv::Mat &my,
                                cv::Mat &Vx,
                                cv::Mat &Vy,
                                cv::Mat &contrast);

        int precomputeParameters(cv::Mat histogram,
                                 std::vector<float> LL,
                                 std::vector<float> AA,
                                 std::vector<float> BB,
                                 int numberOfPixels,
                                 std::vector<int> &reverseMap,
                                 cv::Mat &map,
                                 cv::Mat &colorDistance,
                                 cv::Mat &exponentialColorDistance);

        void calculateHistogram(cv::Mat im,
                                cv::Mat &averageX,
                                cv::Mat &averageY,
                                cv::Mat &averageX2,
                                cv::Mat &averageY2,
                                std::vector<float> &LL,
                                std::vector<float> &AA,
                                std::vector<float> &BB,
                                cv::Mat &histogram,
                                cv::Mat &histogramIndex);



        void outputHowToUse();

        void getSaliencyMap(cv::Mat &inputImage, cv::Mat &saliencyMap);

};
#endif //FASA_H

