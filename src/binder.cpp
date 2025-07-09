#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For automatic std::vector conversions
#include <pybind11/eigen.h>  // For automatic Eigen::Vector conversions

#include "CityEnv.hpp" // Your main simulation header

namespace py = pybind11;

PYBIND11_MODULE(_CityEnvGym, m) {
    m.doc() = "Python bindings for the C++ City Environment Simulator (Single Drone/Target)";

    //================================================
    // 1. Bind Data Structures
    //================================================

    py::class_<city_env::Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("vector", &city_env::Position::vector)
        .def_readwrite("yaw", &city_env::Position::yaw)
        .def("x", &city_env::Position::x)
        .def("y", &city_env::Position::y)
        // Add a __repr__ for easier debugging in Python
        .def("__repr__", [](const city_env::Position &p) {
            return "<Position x=" + std::to_string(p.x()) +
                   ", y=" + std::to_string(p.y()) +
                   ", yaw=" + std::to_string(p.yaw) + ">";
        });

    // Bind the nested Physics struct first
    py::class_<city_env::Drone::Physics>(m, "DronePhysics")
        .def(py::init<>())
        .def_readwrite("mass", &city_env::Drone::Physics::mass)
        .def_readwrite("moment_of_inertia", &city_env::Drone::Physics::moment_of_inertia)
        .def_readwrite("propulsion_gain", &city_env::Drone::Physics::propulsion_gain)
        .def_readwrite("steering_gain", &city_env::Drone::Physics::steering_gain)
        .def_readwrite("linear_drag_coeff", &city_env::Drone::Physics::linear_drag_coeff)
        .def_readwrite("angular_drag_coeff", &city_env::Drone::Physics::angular_drag_coeff);

    py::class_<city_env::Drone>(m, "Drone")
        .def(py::init<>())
        .def_readwrite("id", &city_env::Drone::id)
        .def_readwrite("position", &city_env::Drone::position)
        // Note: The C++ member is 'linear_velocity', but we expose it as 'velocity'
        .def_readwrite("velocity", &city_env::Drone::linear_velocity)
        .def_readwrite("angular_velocity", &city_env::Drone::angular_velocity)
        .def_readwrite("physics", &city_env::Drone::physics);

    py::class_<city_env::Target>(m, "Target")
        .def(py::init<>())
        .def_readwrite("position", &city_env::Target::position)
        .def_readwrite("speed", &city_env::Target::speed)
        .def_readwrite("path", &city_env::Target::path)
        .def_readwrite("current_path_index", &city_env::Target::current_path_index)
        .def_readwrite("radius", &city_env::Target::radius);

    // UPDATED: State now holds a single drone and target
    py::class_<city_env::State>(m, "State")
        .def(py::init<>())
        .def_readwrite("drone", &city_env::State::drone)
        .def_readwrite("target", &city_env::State::target)
        .def_readwrite("time_elapsed", &city_env::State::time_elapsed)
        .def_readwrite("reward", &city_env::State::reward);


    //================================================
    // 2. Bind the Main CityEnv Class
    //================================================

    py::class_<city_env::CityEnv>(m, "CityEnv")
        // UPDATED: The constructor now takes a single drone and target
        .def(py::init<
                const std::vector<std::vector<bool>>&,
                float,
                float,
                float,
                float,
                float,
                city_env::Drone,
                city_env::Target
             >(),
             py::arg("obstacle_map"),
             py::arg("world_width"),
             py::arg("world_height"),
             py::arg("time_step"),
             py::arg("fov_angle"),
             py::arg("fov_distance"),
             py::arg("drone"),
             py::arg("target"),
             "The main constructor for the CityEnv class.")

        .def("reset", &city_env::CityEnv::reset,
             "Resets the environment to its initial state and returns the new state.")

        // UPDATED: The step function now takes a single action
        .def("step", &city_env::CityEnv::step,
             py::arg("action"),
             "Advances the simulation by one time step with a given action and returns the new state.")

        .def("get_state", &city_env::CityEnv::get_state,
             "Returns the current state of the environment without advancing the simulation.")
             
        // UPDATED: checkFov no longer takes arguments and uses internal state
        .def("check_fov", &city_env::CityEnv::checkFov,
            "Checks if the target is within the drone's field of view.")
        .def("map_to_world", &city_env::CityEnv::mapToWorld,
             py::arg("mapCoords"),
             "Converts map coordinates to world coordinates.")
        .def("world_to_map", &city_env::CityEnv::worldToMap,
             py::arg("worldCoords"),
             "Converts world coordinates to map coordinates.");
}