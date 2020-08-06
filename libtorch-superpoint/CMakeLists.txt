cmake_minimum_required(VERSION 3.12)
project(torchtest VERSION 1.0.0)
find_package(Torch REQUIRED)

set(OpenCV_DIR /usr/local/Cellar/opencv/4.4.0/lib/cmake/opencv4) 
find_package(OpenCV REQUIRED)
# add_library(say-hello hello.hpp hello.cpp )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # 让camke找到头文件
# link_directories(${CMAKE_CURRENT_LIST_DIR}/lib) # 让cmake找到库文件
# link_directories(${CMAKE_BINARY_DIR}/bin) 

# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

message(STATUS "find project source dir: ${PROJECT_SOURCE_DIR}")
set (source_dir ${PROJECT_SOURCE_DIR}/src/)
file (GLOB source_files "${source_dir}/*.cpp")
add_executable(a ${source_files})

# message(STATUS "find libtorch in: ${TORCH_LIBRARIES}") # print message
# message(STATUS "find opencv in: ${OpenCV_LIBS}") 
target_link_libraries(a ${TORCH_LIBRARIES} ${OpenCV_LIBS})  # 链接哪个库文件（静态/动态）
set_property(TARGET a PROPERTY CXX_STANDARD 17) 