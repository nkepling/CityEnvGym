#include "CityEnv.hpp"
#include "iostream"
#include <cmath> // Required for std::fmod, std::cos, std::sin
#include "AStar.hpp"


namespace city_env {

    /**
     * @brief Construct a new CityEnv for a single drone and target.
     */
    CityEnv::CityEnv(
        const std::vector<std::vector<bool>>& obstacle_map,
        float world_width,
        float world_height,
        float time_step,
        float fov_angle,
        float fov_distance,
        Drone drone,   
        Target target, 
        float resolution,
        const Eigen::Vector2f& origin,
        const std::vector<Sensor>& sensors
    )
    : obstacle_map(obstacle_map),
      world_width(world_width),
      world_height(world_height),
      time_step(time_step),
      fov_angle(fov_angle),
      fov_distance(fov_distance),
      time_elapsed(0.0f),
      drone(drone),    
      target(target),  
      resolution(resolution),
      origin(origin),
      _sensors(sensors),
      random_generator(std::random_device()()), // Initialize random generator
      angle_distribution(0.0f, 2.0f * M_PI)
    {
        std::cout << "Initializing CityEnv with 1 drone and 1 target." << std::endl;
        this->drone.id = 0; // Set a default ID for the drone
        this->target.position.vector = Eigen::Vector2f(0.0f, 0.0f);
        precompute_target_path();
    }

    /**
     * @brief Resets the environment to its initial state.
     */
    State CityEnv::reset() {
        drone.position.vector.setZero();
        drone.position.yaw = 0.0f;
        drone.linear_velocity.setZero();
        drone.angular_velocity = 0.0f;
        
        target.position.vector.setZero();
        target.position.yaw = 0.0f;

        time_elapsed = 0.0f;

        target.position.vector.setZero();
        precompute_target_path();
        return get_state();
    }

    /**
     * @brief Advances the environment by one time step based on the given action.
     */
    State CityEnv::step(const Action& action) { // Now takes a single Action
        update_drone(action);
        update_target(); // Logic to update target's state if it's dynamic
        check_collision();
        this->time_elapsed += this->time_step;
        return get_state();
    }

    /**
     * @brief Gets the current state of the environment.
     */
    State CityEnv::get_state() const {
        State state;
        state.drone = this->drone;
        state.target = this->target; // Include the target in the state
        state.time_elapsed = this->time_elapsed;
        state.reward = compute_reward(); // Compute the reward based on the current state
        return state;
    }

    float CityEnv::compute_reward() const {
        //pursuit reward 
        Eigen::Vector2f vector_to_target = target.position.vector - drone.position.vector;
        float distance_to_target = vector_to_target.norm(); 
        if (checkFov()) {
            return 10.0f - distance_to_target;
        }

        return -distance_to_target;
    }

    /**
     * @brief Updates the drone's physics based on the provided action.
     */
     void CityEnv::update_drone(const Action& action) {

        const Eigen::Vector2f commanded_linear_velocity = action.head<2>();
        const float commanded_angular_velocity = action[2]; // This is omega
        const auto& physics = drone.physics;

        float steering_torque = physics.steering_gain * (commanded_angular_velocity - drone.angular_velocity);
        float angular_drag_torque = -physics.angular_drag_coeff * drone.angular_velocity;
        float net_torque = steering_torque + angular_drag_torque;
        float angular_acceleration = net_torque / physics.moment_of_inertia;

        drone.angular_velocity += angular_acceleration * this->time_step;
        drone.position.yaw += drone.angular_velocity * this->time_step;
        
        drone.position.yaw = std::fmod(drone.position.yaw + M_PI, 2.0 * M_PI);
        if (drone.position.yaw < 0.0) {
            drone.position.yaw += 2.0 * M_PI;
        }
        drone.position.yaw -= M_PI;

        Eigen::Vector2f commanded_velocity_world_frame;
        commanded_velocity_world_frame.x() = commanded_linear_velocity.x() * std::cos(drone.position.yaw) - commanded_linear_velocity.y() * std::sin(drone.position.yaw);
        commanded_velocity_world_frame.y() = commanded_linear_velocity.x() * std::sin(drone.position.yaw) + commanded_linear_velocity.y() * std::cos(drone.position.yaw);

        Eigen::Vector2f propulsion_force = physics.propulsion_gain * (commanded_velocity_world_frame - drone.linear_velocity);
        Eigen::Vector2f drag_force = -physics.linear_drag_coeff * drone.linear_velocity;
        Eigen::Vector2f net_force = propulsion_force + drag_force;
        Eigen::Vector2f linear_acceleration = net_force / physics.mass;

        drone.linear_velocity += linear_acceleration * this->time_step;
        drone.position.vector += drone.linear_velocity * this->time_step;

    }

    /**
     * @brief Precomputes a "patrolling" path of a specific total distance.
     *
     * This function generates a path where the target moves straight in a
     * random direction until it's about to hit a boundary or an obstacle.
     * It then reverses direction and continues moving straight.
     *
     * @param total_distance The total distance the target should travel in the generated path.
     */
    void CityEnv::precompute_target_path() {
        target.path.clear();
        target.current_path_index = 0;

        Eigen::Vector2i start_node = worldToMap(target.position.vector);

        // --- 1. Find a random, valid, AND DIFFERENT goal position on the grid ---
        Eigen::Vector2i goal_node;
        std::uniform_int_distribution<int> x_dist(0, obstacle_map[0].size() - 1);
        std::uniform_int_distribution<int> y_dist(0, obstacle_map.size() - 1);

        do {
            goal_node.x() = x_dist(random_generator);
            goal_node.y() = y_dist(random_generator);
        } while (obstacle_map[goal_node.y()][goal_node.x()] || goal_node == start_node);

        // --- 2. Use A* to find a path from start_node to goal_node ---

        std::vector<Eigen::Vector2i> path = AStar::findPath(
            start_node,
            goal_node,
            obstacle_map,
            true // Allow diagonal movement
        );

        for (const Eigen::Vector2i& node : path) {
            // Convert each grid node to world coordinates and add to the path
            target.path.push_back(mapToWorld(node));
        }

        std::cout << "Precomputed target path with " << target.path.size() << " waypoints." << std::endl;
    }

    void CityEnv::update_target() {
        if (target.path.empty() || target.current_path_index >= target.path.size() - 1) {
            precompute_target_path();
            if (target.path.size() < 2) return;
        }
        const Eigen::Vector2f& next_waypoint = target.path[target.current_path_index + 1];
        Eigen::Vector2f direction_to_waypoint = (next_waypoint - target.position.vector).normalized();
        float distance_to_move = target.speed * time_step;

        float distance_to_waypoint = (next_waypoint - target.position.vector).norm();
        if (distance_to_move >= distance_to_waypoint) {
            target.position.vector = next_waypoint; 
            target.current_path_index++;
        } else {
            target.position.vector += direction_to_waypoint * distance_to_move;
        }
        target.position.yaw = std::atan2(direction_to_waypoint.y(), direction_to_waypoint.x());
    }


    bool CityEnv::is_in_bounds(const Eigen::Vector2i& grid_pos) const {
        return grid_pos.x() >= 0 && grid_pos.x() < obstacle_map[0].size() &&
               grid_pos.y() >= 0 && grid_pos.y() < obstacle_map.size();
    }


     Eigen::Vector2f CityEnv::mapToWorld(const Eigen::Vector2i& mapCoords) const {
        // Cast the integer map coordinates to float, scale by resolution, and shift by the origin
        return mapCoords.cast<float>() * resolution + origin;
    }

    Eigen::Vector2i CityEnv::worldToMap(const Eigen::Vector2f& worldCoords) const {
        // Shift the world coordinates by the origin, scale down by resolution, and cast to integer
        return ((worldCoords - origin) / resolution).cast<int>();
    }



    /**
     * @brief Checks if the drone has collided with an obstacle or gone out of bounds.
     */
     void CityEnv::check_collision() {
        // UPDATED: Using the new worldToMap function
        Eigen::Vector2i grid_pos = worldToMap(drone.position.vector);

        if (!is_in_bounds(grid_pos)) {
            std::cout << "Drone " << drone.id << " is out of bounds!" << std::endl;
            return;
        }

        if (obstacle_map[grid_pos.y()][grid_pos.x()]) {
            std::cout << "Drone " << drone.id << " collided with an obstacle!" << std::endl;
        }
    }


    bool CityEnv::isLineOfSightClear(const Eigen::Vector2i& start, const Eigen::Vector2i& end) const {
        int x0 = start.x(), y0 = start.y();
        int x1 = end.x(), y1 = end.y();

        int dx = std::abs(x1 - x0);
        int dy = -std::abs(y1 - y0);

        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        
        int err = dx + dy;

        while (true) {
            // Check if the current cell is an obstacle
            if (is_in_bounds({x0, y0}) && obstacle_map[y0][x0]) {
                return false; // Blocked by an obstacle
            }
            if (x0 == x1 && y0 == y1) {
                break; // Reached the end
            }
            
            int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
        return true; // Line of sight is clear
    }

    bool CityEnv::checkFov() const {
        // 1. Check if the target is within the maximum FOV distance
        Eigen::Vector2f vector_to_target = target.position.vector - drone.position.vector;
        if (vector_to_target.squaredNorm() > fov_distance * fov_distance) {
            return false;
        }

        // 2. Check if the target is within the FOV angle
        Eigen::Vector2f drone_forward(std::cos(drone.position.yaw), std::sin(drone.position.yaw));
        vector_to_target.normalize();

        float dot_product = drone_forward.dot(vector_to_target);
        // Clamp dot_product to avoid floating point inaccuracies with acos
        dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
        float angle_to_target = std::acos(dot_product);

        if (std::abs(angle_to_target) > fov_angle / 2.0f) {
            return false;
        }
        
        Eigen::Vector2i drone_grid = worldToMap(drone.position.vector);
        Eigen::Vector2i target_grid = worldToMap(target.position.vector);
        if (!isLineOfSightClear(drone_grid, target_grid)) {
            return false;
        }
        return true;
    }


    bool CityEnv::checkSensors() const {
         if (_sensors.empty()) { 
            return false; 
        }

        for (const auto& sensor : _sensors) {
            Eigen::Vector2f vector_to_target = target.position.vector - sensor.position;
            float distance_to_target = vector_to_target.norm();

            float sensor_radius = sensor.radius; // Assuming each sensor has a radius attribute

            if (distance_to_target < sensor_radius) {

                Eigen::Vector2i target_grid = worldToMap(target.position.vector);
                Eigen::Vector2i sensor_grid = worldToMap(sensor.position);
                if (isLineOfSightClear(sensor_grid, target_grid)) {
                    return true; // Target is detected by this sensor
                }
            }
        }
        return false; 
    }
} 