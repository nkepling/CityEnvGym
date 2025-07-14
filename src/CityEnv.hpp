#pragma once

#include <vector>
#include <utility> // For std::pair
#include <Eigen/Dense>
#include <random> 

namespace city_env {

    // vx,vy,angular_acceleration
    using Action = Eigen::Vector3f;

    struct Sensor {
        Eigen::Vector2f position; // Position of the sensor in world coordinates
        float radius = 10.0f;      // Radius of the sensor's coverage area

        Sensor(float x, float y, float r) : position(x, y), radius(r) {}
        Sensor() : position(0, 0), radius(10.0f) {} // Default constructor
    };

    struct Position {
        Eigen::Vector2f vector; // Represents x, y
        float yaw = 0.0f;       // Heading angle in radians
        float x() const { return vector.x(); }
        float y() const { return vector.y(); }
    };

    struct Drone {
        struct Physics {
            float mass = 1.0f;                  // kg
            float moment_of_inertia = 0.1f;     // kg*m^2 (for rotation)
            float propulsion_gain = 5.0f;       // Proportional gain for linear velocity
            float steering_gain = 2.0f;         // Proportional gain for angular velocity
            float linear_drag_coeff = 0.5f;     // Coefficient for linear drag
            float angular_drag_coeff = 0.1f;    // Coefficient for rotational drag
        };

        int id;
        Position position;
        Eigen::Vector2f linear_velocity;
        float angular_velocity = 0.0f;
        Physics physics;

        Drone() : linear_velocity(0, 0) {} // Default constructor
    };

      struct Target {
        Position position;
        float speed = 10.0f; // How fast the target moves along its path
        int num_steps = 100; // Number of steps to precompute for the target's path

        // Path-following members
        std::vector<Eigen::Vector2f> path;
        float radius = 5.0f; // Radius of the target for collision detection
        int current_path_index = 0;
    };


    // State now holds a single drone and target
    struct State {
        Drone drone;
        Target target;
        float time_elapsed;
        float reward = 0.0f; // Reward can be computed based on the drone's state
    };

    class CityEnv {
    public:
        // Constructor now takes single drone and target objects
        CityEnv(
            const std::vector<std::vector<bool>>& obstacle_map,
            float world_width,
            float world_height,
            float time_step,
            float fov_angle,
            float fov_distance,
            Drone drone,
            Target target,
            float resolution = 1.0f,
            const Eigen::Vector2f& origin = Eigen::Vector2f(-500.0f, -500.0f),
            const std::vector<city_env::Sensor>& sensors = {}
        );
        
        State reset();
        // Step now takes a single action
        State step(const Action& action);
        State get_state() const;
        float compute_reward() const;   

        bool isLineOfSightClear(const Eigen::Vector2i& start, const Eigen::Vector2i& end) const;    
        // checkFov now checks the internal drone against the internal target
        bool checkFov() const;
        bool checkSensors() const;
        Eigen::Vector2f mapToWorld(const Eigen::Vector2i& mapCoords) const;
        Eigen::Vector2i worldToMap(const Eigen::Vector2f& worldCoords) const;

    private:
        // Member variables updated for a single drone/target
        Drone drone;
        Target target;
        
        std::vector<std::vector<bool>> obstacle_map;
        float world_width;
        float world_height;
        float time_step;
        float fov_angle;
        float fov_distance;
        float time_elapsed;
        float resolution;
        Eigen::Vector2f origin;
        std::vector<city_env::Sensor> _sensors;

        void update_drone(const Action& action);
        void update_target();
        void check_collision();
        void reset_drone();
        void reset_target();
        bool is_in_bounds(const Eigen::Vector2i& grid_pos) const;
        mutable std::mt19937 random_generator;
        std::uniform_real_distribution<float> angle_distribution;
        void precompute_target_path();
    };
} // namespace city_env