cmake_minimum_required(VERSION 3.15)
project(SLIDE_DML)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(SLIDE_DML
        Connection.h
        main.cpp
        Net.cpp
        Net.h
        Neuron.cpp
        Neuron.h)
