
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)
project(reduceSumCuda LANGUAGES CXX CUDA)

add_library(reduceSumLib STATIC
    reduceSum.cu
)
add_library(legacyLib STATIC
    legacy.cu
)

add_executable(reduceSumCuda ./testReduceSum.cc)
find_package(CUDA)
target_include_directories(reduceSumCuda PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(reduceSumCuda PRIVATE
    reduceSumLib
    legacyLib
)
