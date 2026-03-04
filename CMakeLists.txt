cmake_minimum_required(VERSION 3.15)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(financeARLM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/nnets)
include_directories(${PROJECT_SOURCE_DIR}/nnets/networks)
include_directories(${PROJECT_SOURCE_DIR}/nnets/optimizers)
include_directories(${PROJECT_SOURCE_DIR}/nnets/LRSchedulers)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Include Eigen (adjust this path to where you put Eigen in WSL)
# include_directories("~/eigen/eigen-5.0.0")

set(COMMON_MODULES
    src/tensor.cpp
    src/autograd.cpp
    src/MatrixMultiply.cpp
    src/nditerator.cpp
)

#add module
pybind11_add_module(peped bindings.cpp ${COMMON_MODULES})
target_compile_options(peped PRIVATE -O3 -mavx2 -mfma)
target_link_libraries(peped
    PRIVATE
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)


# Add executable
add_executable(financeARLM src/main.cpp ${COMMON_MODULES})
target_compile_options(financeARLM PRIVATE -O3 -mavx2 -mfma)
target_link_libraries(financeARLM 
    PRIVATE 
    OpenMP::OpenMP_CXX 
    ${BLAS_LIBRARIES} 
    ${LAPACK_LIBRARIES}
)

    
install(TARGETS peped DESTINATION .)

# Testing cmake lists
option(BUILD_TESTS "Test Builds" OFF)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_files
        test_files/autograd_test.cpp
        ${COMMON_MODULES}    
    )
    target_compile_options(test_files PRIVATE -O0 -g)
    target_link_libraries(test_files
        PRIVATE
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
    )

    add_test(NAME test_files COMMAND test_files)
endif()
