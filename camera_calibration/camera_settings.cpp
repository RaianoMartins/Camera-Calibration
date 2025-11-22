#include "camera_settings.hpp"
#include <iostream>

void CalibrationSettings::write(cv::FileStorage& fs) const
{
    fs << "{"
       << "BoardSizeWidth"              << _boardSize.width
       << "BoardSizeHeight"             << _boardSize.height
       << "SquareSize"                  << _square_size
       << "CalibrateNrOfFrameToUse"     << _n_frames
       << "InputDelay"                  << _delay
       << "}";

    return;
}

void CalibrationSettings::read(const cv::FileNode& node)
{
    node["BoardSizeWidth"]          >> _boardSize.width;
    node["BoardSizeHeight"]         >> _boardSize.height;
    node["SquareSize"]              >> _square_size;
    node["CalibrateNrOfFrameToUse"] >> _n_frames;
    node["InputDelay"]              >> _delay;
    node["CameraID"]                >> _camera_id;
    node["OutputFile"]              >> _output_file_name; 
    node["CameraResolutionWitdh"]   >> _video_width;
    node["CameraResolutionHeight"]  >> _video_height; 
    node["VideoFrameRate"]          >> _frame_rate;
    node["CameraAutoExposure"]      >> _auto_exposure;
    validate();
}

void CalibrationSettings::validate()
{
    valid_input = true;

    if (_boardSize.width <= 0 || _boardSize.height <= 0)
    {
        std::cerr << "Invalid Board size: "
                  << _boardSize.width << " x " << _boardSize.height << std::endl;
        valid_input = false;
    }

    if (_square_size <= 1e-6)
    {
        std::cerr << "Invalid square size: " << _square_size << std::endl;
        valid_input = false;
    }

    if (_n_frames <= 0)
    {
        std::cerr << "Invalid number of frames: " << _n_frames << std::endl;
        valid_input = false;
    }
}

float averageCornerDistance(std::vector<cv::Point2f>& vec_a, std::vector<cv::Point2f>& vec_b)
{
    float sum = 0;

    for(int i = 0; i < vec_a.size(); i++)
    {
        float dx = vec_a[i].x - vec_b[i].x;
        float dy = vec_a[i].y - vec_b[i].y;
        sum = std::sqrt(dx*dx + dy*dy);
    }

    return sum / vec_a.size();
}
