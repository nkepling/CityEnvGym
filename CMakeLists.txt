cmake_minimum_required(VERSION 3.15...3.30)
project(CityEnvGym)

# --- Dependencies ---
include(FetchContent)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Eigen3 REQUIRED)
add_subdirectory(pybind11)


FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
)


find_package(Eigen3 QUIET)

if(NOT Eigen3_FOUND)
    FetchContent_MakeAvailable(Eigen3)
endif()


add_library(CityEnv STATIC
    src/CityEnv.cpp
    src/AStar.cpp
    src/MultiAgentCityEnv.cpp
)

target_compile_features(CityEnv PUBLIC cxx_std_17)
set_target_properties(CityEnv PROPERTIES POSITION_INDEPENDENT_CODE ON)



target_link_libraries(CityEnv PUBLIC
    Eigen3::Eigen
)


pybind11_add_module(_CityEnvGym src/binder.cpp)
target_link_libraries(_CityEnvGym PRIVATE CityEnv)

install(TARGETS _CityEnvGym DESTINATION CityEnvGym)

add_executable(city_env_test src/main.cpp)
target_link_libraries(city_env_test PRIVATE CityEnv)