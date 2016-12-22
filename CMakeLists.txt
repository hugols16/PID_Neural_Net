cmake_minimum_required(VERSION 3.4.1)
project(PIDNN_SIM)
include_directories(include)
add_executable(PIDNN_SIM src/PID_Neural_Net_Bundle.cpp src/PIDNN.cpp)