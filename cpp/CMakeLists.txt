cmake_minimum_required(VERSION 3.25)
project(
        cpp
        VERSION 0.1.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Looks for pybind11
find_package(pybind11 REQUIRED)

# C++ files !! No main.cpp !!
set(CORE_FILES
        src/binomial_pricer.cpp
        src/convergence_results.cpp
        src/trinomial_pricer.cpp
        # include/convergence_results.h
        src/functions.cpp
)

# =====================================================================
# TARGET 1 : PYTHON MODULE (discrete_models)
# =====================================================================
pybind11_add_module(discrete_models
        src/bindings.cpp            # pybind11 logic
        ${CORE_FILES}          # C++ Core files
)

target_include_directories(discrete_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Stating binding (Anti DLL failed)
if(WIN32 AND NOT MSVC)
    target_link_options(discrete_models PRIVATE
            "-static-libgcc"
            "-static-libstdc++"
            "-Wl,-Bstatic"
            "-lwinpthread"
            "-Wl,-Bdynamic"
    )
endif()

# Auto copy in Python .venv
set_target_properties(discrete_models PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "D:/01b-Dev/Python/Projects/Discrete_models/.venv/Lib/site-packages"
)

# =====================================================================
# TARGET 2 : TEST LAB C++ (cpp)
# =====================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(cpp
            src/main.cpp
            ${CORE_FILES}
    )
    target_include_directories(cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()