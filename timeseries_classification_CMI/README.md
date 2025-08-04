# BFRB Detection from Wrist Sensor Data (Child Mind Institute Competition)

## 📌 Overview

This project is my first time series classification project, developed for the Child Mind Institute's competition on detecting Body-Focused Repetitive Behaviors (BFRBs). The goal is to build a model that distinguishes BFRB-like gestures (e.g., hair pulling, nail biting) from everyday non-BFRB gestures (e.g., adjusting glasses), using multimodal sensor data collected from a custom wrist-worn device called **Helios**.

## 🔗 Competition Link
[Click here to go to the competition site](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/overview)


Helios includes the following sensor types:
- **Inertial Measurement Unit (IMU)** — acceleration and rotation data
- **Thermopiles** — skin temperature sensing
- **Time-of-Flight (ToF)** — proximity sensing via distance measurements

Each gesture sequence includes a **transition**, **pause**, and **gesture**, recorded with the device worn on the **dominant wrist**.

## 🛠️ My Approach

This solution is built using the [`sktime`](https://www.sktime.org/en/stable/) library for time series classification.

- **Modeling Strategy**: An ensemble of sktime-compatible classifiers was used to capture different aspects of the signal patterns.

## 🧠 Future Improvements

To further improve performance and generalization, I plan to:
- **Incorporate demographic metadata** such as:
  - **Handedness**
  - **Arm length** or body size (could impact ToF and IMU readings)
- **Explore additional base models** in `sktime`, including deep learning approaches.
- **Tune hyperparameters** for individual classifiers in the ensemble.
- **Visualize sensor patterns** for gesture types to improve feature engineering and model understanding.

