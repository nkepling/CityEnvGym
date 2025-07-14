cmake_minimum_required(VERSION 3.15...3.30)
project(CityEnvGym)

# --- Dependencies ---
include(FetchContent)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Eigen3 REQUIRED)
add_subdirectory(pybind11)


add_library(CityEnv STATIC
    src/CityEnv.cpp
    src/AStar.cpp
)

set_target_properties(CityEnv PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 4. Link CityEnv against the A* library and other dependencies.
target_link_libraries(CityEnv PUBLIC
    Eigen3::Eigen
)

## Final Targets

# Python Module (no changes needed here)
pybind11_add_module(_CityEnvGym src/binder.cpp)
target_link_libraries(_CityEnvGym PRIVATE CityEnv)

install(TARGETS _CityEnvGym DESTINATION CityEnvGym)

# Test Executable (no changes needed here)
add_executable(city_env_test src/main.cpp)
target_link_libraries(city_env_test PRIVATE CityEnv)