cmake_minimum_required(VERSION 3.5)
PROJECT(RS_with_CM VERSION 0.10)

SET(SOURCES ./src/main.cpp ./src/gf256.cpp ./src/cm256.cpp)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-msse4.2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

ADD_EXECUTABLE( ${PROJECT_NAME} ${SOURCES} )

IF (CMAKE_BUILD_TYPE STREQUAL DEBUG)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_include_directories(RS_with_CM PUBLIC ./include)