cmake_minimum_required(VERSION 4.3)
project(
        cpp
        VERSION 0.1.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
        cpp
        src/main.cpp
        include/binomial_pricer.h
        src/binomial_pricer.cpp
)

target_include_directories(
        cpp
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)