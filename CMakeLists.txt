cmake_minimum_required(VERSION 2.8)
project( engcv )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( engcv src/main.cpp )

target_link_libraries( engcv ${OpenCV_LIBS} )

