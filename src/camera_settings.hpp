#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define REQUIRED_MS 2000
#define MIN_MOVEMENT 1.0

enum Mode { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

class CalibrationSettings
{
    public:
        CalibrationSettings() : valid_input(false) {}

        void validate();
        void write(cv::FileStorage& fs) const;
        void read(const cv::FileNode& node);

    public:
        float _square_size;              // Tamanho do quadrado
        int _n_frames;                   // Número de frames para calibração
        int _delay;                      // Delay no modo vídeo
        int _camera_id;
        int _video_width;
        int _video_height; 
        int _frame_rate;
        int _auto_exposure;
        cv::Size _boardSize;             // Número de quadrados (largura x altura)
        cv::VideoCapture inputCapture;
        std::string _output_file_name;   // Arquivo de saída
        bool valid_input;
};

float averageCornerDistance(std::vector<cv::Point2f>&, std::vector<cv::Point2f>&);

static inline void read (const cv::FileNode& node, CalibrationSettings& settings, const CalibrationSettings& defaultSettings = CalibrationSettings())
{
    if (node.empty())
        settings = defaultSettings;
    else
        settings.read(node);
}