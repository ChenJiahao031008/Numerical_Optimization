cmake_minimum_required(VERSION 3.10.2)
project(AMao LANGUAGES CXX VERSION 0.0.1)

# -fPIC：生成动态库，-fopenmp 开启多线程，-O3 对代码进行优化，-g 打印调试信息，-Wall 打印所有警告信息, pthread 支持多线程
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17 -g -O3 -fopenmp -pthread")

# 不生成警告信息: debug阶段尽量不要使用
add_definitions(-w)

option(BUILD_TEST "Build all tests." ON)
if(BUILD_TEST)
  find_package(GTest REQUIRED)
  enable_testing()
endif()

find_package(Boost REQUIRED COMPONENTS thread filesystem program_options system)

link_directories(
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/modules/base
    ${PROJECT_SOURCE_DIR}/modules/unconstrained_optimization
)

add_subdirectory(common)
add_subdirectory(modules/base)
add_subdirectory(modules/unconstrained_optimization)



if(BUILD_TEST)
    add_subdirectory(unit_testing)
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE
    # dw
    common
    base
    unconstrained_optimization
    ${Boost_LIBRARIES}
)
