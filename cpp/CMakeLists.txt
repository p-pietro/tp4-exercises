cmake_minimum_required(VERSION 3.10)
project(tp4_exercises)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GSL_INCLUDE_DIRS})

add_executable(main main.cpp functions.cpp)

target_link_libraries(main ${GSL_LIBRARIES} ${GSL_CBLAS_LIBRARIES} Matplot++::matplot)
