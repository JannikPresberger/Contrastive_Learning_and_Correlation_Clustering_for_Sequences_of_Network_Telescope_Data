cmake_minimum_required(VERSION 3.15)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(andres_graph)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(GUROBI REQUIRED)

find_library(
  GUROBI_CXX_LIBRARY
  NAMES gurobi_c++ gurobi_g++8.5
  HINTS ${GUROBI_HOME} $ENV{GUROBI_HOME} ${GUROBI_HOME}/lib $ENV{GUROBI_HOME}/lib
  PATH_SUFFIXES lib
  REQUIRED
)

include_directories(${GUROBI_INCLUDE_DIRS})

pybind11_add_module(andres_graph python-interface.cxx)

if(CXX)
	#set(CMAKE_CXX_STANDARD 11)
	target_link_libraries(${PROJECT_NAME} PRIVATE optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})

target_include_directories(andres_graph PRIVATE include)
target_include_directories(andres_graph PRIVATE bindings)
target_link_libraries(andres_graph PRIVATE ${GUROBI_LIBRARIES} )
target_compile_options(andres_graph PRIVATE -O3 -march=native -ffast-math)

message(STATUS "GUROBI_CXX_LIBRARY: ${GUROBI_CXX_LIBRARY}")
message(STATUS "GUROBI_LIBRARY: ${GUROBI_LIBRARY}")

install(TARGETS andres_graph DESTINATION .)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/andres_graph.pyi")
    install(FILES andres_graph.pyi DESTINATION .)
else()
    message(STATUS "Note: andres_graph.pyi not found, skipping stub installation.")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")