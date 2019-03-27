cmake_minimum_required(VERSION 3.5)
set(KEYWORD saliency_fasa)
set(CMAKE_CXX_STANDARD 11)
project($KEYWORD)

FIND_PACKAGE(YARP REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(ICUBcontrib REQUIRED)


list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(YarpInstallationHelpers)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)


icubcontrib_set_default_prefix()
set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")

#search for includes
INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/include
        ${YARP_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

# Search for source code.
FILE(GLOB folder_source src/*.cpp src/*.cc src/*.c )
FILE(GLOB folder_header include/iCub/*.h  )
SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")

# Set up our main executable
IF (folder_source)
    ADD_EXECUTABLE(${KEYWORD}
            ${folder_source}
            ${folder_header}
            )

    TARGET_LINK_LIBRARIES( ${KEYWORD}
            ${YARP_LIBRARIES}
            ${OpenCV_LIBS}
            )

    install(TARGETS  ${KEYWORD} DESTINATION bin)


ELSE (folder_source)
    MESSAGE( "No source code files found. Please add something")
ENDIF (folder_source)


icubcontrib_add_uninstall_target()