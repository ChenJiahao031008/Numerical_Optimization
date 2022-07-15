project (unconstrained_optimization)

file(GLOB _SRCS "*.cc" "*.h" "*.hpp")
add_library(${PROJECT_NAME} OBJECT ${_SRCS})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    common
    base
)

