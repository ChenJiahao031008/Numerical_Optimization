project(unit_testing)

# 输出可执行文件到指定目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)

add_executable(opt_sgd_test opt_sgd_test.cpp)
target_compile_features(opt_sgd_test PRIVATE cxx_std_17)
target_link_libraries(opt_sgd_test
    gtest gtest_main
    common
    unconstrained_optimization
    ${Boost_LIBRARIES}
)
add_test(NAME opt_sgd_test COMMAND opt_sgd_test)
