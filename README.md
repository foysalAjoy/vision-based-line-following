# Vision-Based Line Following

A robotics and computer vision project investigating how an autonomous differential-drive robot can follow a curved line using camera-based perception.

The project was developed in Webots using an e-puck robot and compares classical image-processing approaches with CNN-based visual error estimation. A key focus is evaluating how the different perception methods behave when illumination conditions change.

## Project Overview

Traditional line-following robots often rely on dedicated ground sensors. In this project, the robot instead uses a downward-facing camera to observe the track.

The system follows the pipeline:

Camera image → Line perception → Tracking error → P/PD controller → Wheel velocities

Three main approaches are evaluated:

- Classical vision with P control
- Classical vision with PD control
- CNN-based line-error estimation with PD control

The same control framework is used where possible so that the effect of changing the perception method can be studied more clearly.

## Classical Vision Pipeline

The classical perception system processes the camera image using a fixed image-processing pipeline:

1. Select the lower region of interest (ROI).
2. Convert the image to grayscale.
3. Apply a fixed threshold to identify dark track pixels.
4. Calculate the centroid of the detected line.
5. Compare the centroid with the image centre.
6. Convert the displacement into a normalized tracking error.

The tracking error is then passed to either a P or PD controller.

### P Controller

The proportional controller produces a steering correction based on the current tracking error.

A larger error produces a larger steering correction.

### PD Controller

The PD controller also considers how quickly the tracking error is changing.

This helps reduce oscillation and improves the robot's response when entering or leaving curved sections of the track.

## CNN-Based Perception

I also investigated learning-based visual perception using a convolutional neural network.

An early version attempted to predict steering commands directly from camera images. Although the network achieved promising offline validation results, its closed-loop behaviour was poor and it tended to predict steering values close to zero on important curved sections.

This highlighted an important difference between offline prediction accuracy and actual closed-loop robotic performance.

The architecture was therefore redesigned.

Instead of predicting wheel steering directly, the final CNN estimates the visual line-position error from the camera image.

The control pipeline became:

Camera image → CNN line-error estimate → PD controller → Wheel velocities

This separates perception from control and allows the learned perception system to use the same control logic as the classical approach.

The trained model is included in:

`models_trained/line_error_cnn.keras`

## Dataset

Training data was collected in Webots using controlled robot runs.

The dataset contains camera frames together with line-error labels. Additional examples were generated to improve the representation of different line positions and recovery situations.

The public repository includes dataset manifests, summaries and representative analysis outputs.

The full generated image dataset is intentionally excluded from Git because it contains a large number of intermediate training images.

## Robustness Testing

The project also evaluates the perception systems under changing illumination conditions.

This is important because the classical detector relies on a fixed grayscale threshold. As illumination changes, the contrast between the line and surrounding surface can change significantly.

The robustness experiments compare how the classical and CNN-based approaches behave as the simulated lighting conditions are varied.

Evaluation includes:

- track completion
- line visibility
- cross-track error
- mean absolute error (MAE)
- root mean squared error (RMSE)
- maximum tracking error
- overshoot
- settling behaviour
- robustness under illumination variation

## Results and Analysis

Processed experiment summaries and figures are available in:

`evaluation_summary_final/`

This includes:

- method comparison results
- CNN validation results
- predicted vs. ground-truth plots
- CNN error distributions
- training curves
- robustness sweeps
- transient-response analysis
- error-over-time plots
- diagnostic results

Additional classical controller comparison figures are available in:

`results/steering_plots/`

## Repository Structure

```text
vision-based-line-following/
├── common/                     Shared controller and vision utilities
├── config/                     Project configuration
├── controllers/                Webots robot controllers
│   ├── p_line_follower/
│   ├── pd_line_follower/
│   ├── cnn_pd_hybrid_final/
│   └── dataset_*/
├── data/                       Dataset manifests and analysis
├── evaluation_summary_final/   Processed experiment results
├── models_trained/             Trained CNN and training history
├── protos/                     Webots robot PROTO files
├── results/                    Selected plots and result tables
├── scripts/                    Training, evaluation and analysis tools
├── worlds/                     Webots simulation environments
├── archive/                    Earlier experimental approaches
├── requirements.txt
└── SETUP.md

```

## Main Controllers

### Classical P

`controllers/p_line_follower/p_line_follower.py`

Uses classical image processing to estimate the line position and proportional control to steer the robot.

### Classical PD

`controllers/pd_line_follower/pd_line_follower.py`

Uses the same classical perception pipeline with proportional-derivative control.

### CNN + PD

`controllers/cnn_pd_hybrid_final/cnn_pd_hybrid_final.py`

Uses the trained CNN to estimate line-position error and a PD controller to generate wheel commands.

## Running the Project

### Requirements

The project was developed using:

- Python 3.12
- Webots
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Pillow

Install the Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Then check the environment with:

```bash
python3 scripts/check_environment.py
```

Detailed setup instructions for macOS and Windows are available in [SETUP.md](SETUP.md).

## Running in Webots

Open:

`worlds/line_following_curve_test.wbt`

The project contains separate Webots controllers for the classical P, classical PD and CNN+PD approaches.

Additional worlds are included for dataset collection, baseline experiments and illumination robustness testing.

## Reproducing the Analysis

The `scripts/` directory contains utilities for dataset preparation and auditing, CNN training and validation, experiment generation, robustness testing, transient-response analysis, diagnostics, and generation of result figures and tables.

Run:

```bash
python3 scripts/check_environment.py
```

first to verify that the expected model, dataset metadata and project files are available.

## Engineering Lessons

One of the most useful findings from this project was that good offline neural-network performance does not automatically guarantee good robotic control.

The first end-to-end CNN produced reasonable validation results but did not provide reliable steering when placed inside the closed control loop.

I therefore redesigned the system so that the CNN estimated a physically meaningful line-position error, while a conventional PD controller handled steering. This produced a more interpretable architecture and made it possible to compare learned and classical perception within a similar control framework.

The project demonstrates model development, system integration, debugging, experimental evaluation and iterative robotics design.

## Technologies

**Robotics:** Webots, e-puck, differential-drive control  
**Computer Vision:** OpenCV, image thresholding, centroid detection  
**Machine Learning:** TensorFlow, CNN regression  
**Control:** P and PD control  
**Programming:** Python  
**Analysis:** NumPy, Pandas, Matplotlib

## Author

**Md Foysal Ahammad Joy**

MSc Robotics & Artificial Intelligence  
University of Aberdeen

GitHub: `foysalAjoy`
