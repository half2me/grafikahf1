cmake_minimum_required(VERSION 3.6)
project(grafika)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS}  $(GLEW_INCLUDE_DIRS))

add_executable(grafika ${SOURCE_FILES})

target_link_libraries(grafika ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY})