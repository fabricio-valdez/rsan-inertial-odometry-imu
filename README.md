# Inertial Odometry using IMU

This project implements an inertial odometry system using walking data that was previously collected with an IMU ROS2-based package I created. This project demonstrates why navigation using IMU data alone is a hard problem! 

Course: Robotics Sensing and Navigation (RSAN)

## Tools
Hardware: Vectornav VN-100 IMU
Data sets: One data set of walking in a circle
           One data set of walking in a square
           
## Objectives
- Process and analyze inertial sensor data collected from a VectorNav IMU during walking experiments.
- Calibrate magnetometer measurements using circular walking data to correct distortions in the magnetic field readings.
- Estimate orientation by integrating gyroscope rotational rates and compare the results with heading estimates derived from magnetometer measurements.
- Estimate velocity and displacement by integrating accelerometer measurements and analyze the drift introduced by inertial integration.
- Reconstruct 2D trajectories (North vs. East position) using heading estimates and integrated acceleration data.
- Compare trajectory estimation using gyro-based heading and magnetometer-based heading.
- Evaluate the challenges and limitations of inertial odometry when relying only on IMU measurements.

## Methods
- Magnetometer calibration
- Gyroscope integration
- Acceleration integration
- Heading estimation
- Trajectory reconstruction

## Results
Results can be found in the Results Report.


*This project comes from an assignment I did for the class of Robot Sensing and Navigation.
