#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "camera_settings.hpp"

int main()
{
    // === Carregar configurações ===
    CalibrationSettings settings;
    cv::FileStorage fs("config.xml", cv::FileStorage::READ);
    fs["CalibrationSettings"] >> settings;
    fs.release();

    if (!settings.valid_input) 
    {
        std::cerr << "Configuração inválida!\n";
        return -1;
    }

    cv::VideoCapture cap(settings._camera_id, cv::CAP_V4L2);
    if (!cap.isOpened()) 
    {
        std::cerr << "Erro ao abrir a câmera.\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  settings._video_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, settings._video_height);
    cap.set(cv::CAP_PROP_FPS,          settings._frame_rate);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, settings._auto_exposure);  

    std::vector<cv::Point3f> obj_p;
    obj_p.reserve(settings._boardSize.area());

    for (int i = 0; i < settings._boardSize.height; i++)
        for (int j = 0; j < settings._boardSize.width; j++)
            obj_p.emplace_back(j * settings._square_size, i * settings._square_size, 0);

    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;

    Mode mode = DETECTION;
    int captured = 0;

    bool counting = false;
    bool has_last = false;

    std::chrono::steady_clock::time_point t_start;
    cv::Mat frame, gray;
    std::vector<cv::Point2f> corners, last_corners;

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 80, 1e-5);
    const int corner_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FILTER_QUADS;
    const int calibrate_flags = cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_ASPECT_RATIO; 

    while (true)
    {
        if (!cap.read(frame)) 
        {
            std::cerr << "Frame vazio.\n";
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (mode == CAPTURING)
        {
            bool found = cv::findChessboardCorners(gray, settings._boardSize, corners, corner_flags);

            if (found)
            {
                cv::cornerSubPix(gray, corners, {11,11}, {-1,-1}, criteria);
                cv::drawChessboardCorners(frame, settings._boardSize, corners, true);

                if (!counting)
                {
                    counting = true;
                    t_start = std::chrono::steady_clock::now();
                }

                unsigned ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();

                bool moved = true;
                if (has_last)
                {
                    float dist = averageCornerDistance(corners, last_corners);
                    moved = (dist > MIN_MOVEMENT);

                    if (!moved)
                        cv::putText(frame, "Mova o tabuleiro!", {10,60}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,255}, 2);
                }

                if (moved && ms >= REQUIRED_MS)
                {
                    obj_points.push_back(obj_p);
                    img_points.push_back(corners);
                    captured++;

                    std::cout << "Capturado " << captured << "/" << settings._n_frames << std::endl;

                    last_corners = corners;
                    has_last = true;
                    counting = false;

                    if (captured >= settings._n_frames)
                    {
                        mode = CALIBRATED;
                        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                        camera_matrix.at<double>(0,0) = 1.0; 
                        camera_matrix.at<double>(1,1) = 1.0;
                        cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
                        std::vector<cv::Mat> rvecs, tvecs;

                        double err = cv::calibrateCamera( obj_points, img_points, frame.size(), camera_matrix, dist_coeffs, rvecs, tvecs, calibrate_flags, criteria);

                        std::cout << "\n==== CALIBRAÇÃO COMPLETA ====\n";
                        std::cout << "Erro médio de reprojeção: " << err << "\n";
                        std::cout << "Matriz da câmera:\n" << camera_matrix << "\n";
                        std::cout << "Coeficientes de distorção:\n" << dist_coeffs << "\n";

                        cv::FileStorage fs2("calibration_output.xml", cv::FileStorage::WRITE);
                        fs2 << "camera_matrix" << camera_matrix;
                        fs2 << "dist_coeffs"   << dist_coeffs;
                        fs2.release();
                    }
                }
            }
        }

        // === Mensagem de Status ===
        std::string msg =
            (mode == DETECTION)  ? "Pressione 'g' para iniciar captura" :
            (mode == CAPTURING)  ? "Capturando (" + std::to_string(captured) + "/" + std::to_string(settings._n_frames) + ")" :
                                   "Calibrado!";

        cv::putText(frame, msg, {10,30}, cv::FONT_HERSHEY_SIMPLEX,
                    0.8, {0,255,0}, 2);

        cv::imshow("Calibracao da camera", frame);

        // === Controles ===
        char key = cv::waitKey(1);
        if (key == 27) break; // ESC

        if (key == 'g' && mode == DETECTION)
        {
            std::cout << "Iniciando captura...\n";
            mode = CAPTURING;
        }
    }

    return 0;
}
