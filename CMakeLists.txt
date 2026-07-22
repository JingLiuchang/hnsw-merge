cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" ) # debug
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" ) # release
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" ) # debug
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" ) # release
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "-DHAVE_CXX0X /W1 /openmp /EHsc" ) # debug
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" ) # release
    endif()

    # 添加 TBB 支持
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TBB tbb)
    endif()

    if(NOT TBB_FOUND)
        find_library(TBB_LIBRARY tbb)
        if(TBB_LIBRARY)
            set(TBB_LIBRARIES ${TBB_LIBRARY})
            message(STATUS "Found TBB library: ${TBB_LIBRARY}")
        else()
            message(WARNING "TBB library not found. Some features may not work.")
            set(TBB_LIBRARIES "")
        endif()
    else()
        message(STATUS "Found TBB via pkg-config")
    endif()

    # examples
    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search hnswlib)

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search hnswlib)

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search hnswlib)

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter hnswlib)

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted hnswlib)

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search hnswlib)

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter hnswlib)

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted hnswlib)

    add_executable(test_NGM_merge examples/cpp/test_NGM_merge.cpp)
    target_link_libraries(test_NGM_merge hnswlib ${TBB_LIBRARIES})

    add_executable(test_RGTM_merge examples/cpp/test_RGTM_merge.cpp)
    target_link_libraries(test_RGTM_merge hnswlib ${TBB_LIBRARIES})

    add_executable(test_NSM_merge examples/cpp/test_NSM_merge.cpp)
    target_link_libraries(test_NSM_merge hnswlib ${TBB_LIBRARIES})

    add_executable(test_SIM_merge examples/cpp/test_SIM_merge.cpp)
    target_link_libraries(test_SIM_merge hnswlib ${TBB_LIBRARIES})

    add_executable(test_OVERLAP_merge examples/cpp/test_OVERLAP_merge.cpp)
    target_link_libraries(test_OVERLAP_merge hnswlib ${TBB_LIBRARIES})

    add_executable(test_hnsw_index examples/cpp/test_hnsw_index.cpp)
    target_link_libraries(test_hnsw_index hnswlib ${TBB_LIBRARIES})

    add_executable(test_hnsw_level0_index examples/cpp/test_hnsw_level0_index.cpp)
    target_link_libraries(test_hnsw_level0_index hnswlib ${TBB_LIBRARIES})

    add_executable(test_hnsw_search examples/cpp/test_hnsw_search.cpp)
    target_link_libraries(test_hnsw_search hnswlib ${TBB_LIBRARIES})

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test hnswlib)

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test hnswlib)

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test hnswlib)

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test hnswlib)

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test hnswlib)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main hnswlib)
endif()
