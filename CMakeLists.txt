cmake_minimum_required(VERSION 2.8)
project( engcv )
find_package( PkgConfig REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
pkg_search_module( TESSERACT REQUIRED tesseract )

pkg_search_module( LEPTONICA REQUIRED lept )

include_directories( ${TESSERACT_INCLUDE_DIRS} )

include_directories( ${LEPTONICA_INCLUDE_DIRS} )

link_directories( ${TESSERACT_LIBRARY_DIRS} )

link_directories( ${LEPTONICA_LIBRARY_DIRS} )


add_executable( engcv src/main.cpp )

target_link_libraries( engcv ${TESSERACT_LIBRARIES} )
target_link_libraries( engcv ${LEPTONICA_LIBRARIES} )
target_link_libraries( engcv ${OpenCV_LIBS} )

