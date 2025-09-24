#include "CityEnv.hpp"
#include "iostream"
#include <cmath> // Required for std::fmod, std::cos, std::sin
#include "AStar.hpp"
#include <algorithm>
#include <optional> // Required for std::optional


namespace city_env {


    State MultiAgentCityEnv::step(const Action& pursuer_action, const Action& evader_action) {
        update_drone(pursuer_action);
        update_target_with_action(evader_action); 
        check_collision();
        this->time_elapsed += this->time_step;
        return get_state();

    }

    State MultiAgentCityEnv::default_step(const Action& pursuer_action) {
        update_drone(pursuer_action);
        update_target(); 
        check_collision();
        this->time_elapsed += this->time_step;
        return get_state();
    }

    void MultiAgentCityEnv::update_target_with_action(const Action& action) {
        const Eigen::Vector2f commanded_linear_velocity = action.head<2>();
        const float commanded_angular_velocity = action[2]; // This is omega
        const auto& physics = target.physics;

        float steering_torque = physics.steering_gain * (commanded_angular_velocity - target.angular_velocity);
        float angular_drag_torque = -physics.angular_drag_coeff * target.angular_velocity;
        float net_torque = steering_torque + angular_drag_torque;
        float angular_acceleration = net_torque / physics.moment_of_inertia;

        target.angular_velocity += angular_acceleration * this->time_step;
        target.position.yaw += target.angular_velocity * this->time_step;
        
        target.position.yaw = std::fmod(target.position.yaw + M_PI, 2.0 * M_PI);
        if (target.position.yaw < 0.0) {
            target.position.yaw += 2.0 * M_PI;
        }
        target.position.yaw -= M_PI;

        Eigen::Vector2f commanded_velocity_world_frame;
        commanded_velocity_world_frame.x() = commanded_linear_velocity.x() * std::cos(target.position.yaw) - commanded_linear_velocity.y() * std::sin(target.position.yaw);
        commanded_velocity_world_frame.y() = commanded_linear_velocity.x() * std::sin(target.position.yaw) + commanded_linear_velocity.y() * std::cos(target.position.yaw);

        Eigen::Vector2f propulsion_force = physics.propulsion_gain * (commanded_velocity_world_frame - target.linear_velocity);
        Eigen::Vector2f drag_force = -physics.linear_drag_coeff * target.linear_velocity;
        Eigen::Vector2f net_force = propulsion_force + drag_force;
        Eigen::Vector2f linear_acceleration = net_force / physics.mass;

        target.linear_velocity += linear_acceleration * this->time_step;
        target.position.vector += target.linear_velocity * this->time_step;

        // Keep the target within the world boundaries
        // Keep the target within the world boundaries using std::min and std::max
        target.position.vector.x() = std::max(-world_width / 2, std::min(target.position.vector.x(), world_width / 2));
        target.position.vector.y() = std::max(-world_height / 2, std::min(target.position.vector.y(), world_height / 2));

        // clamp max speed and angular velocity
        if (target.linear_velocity.norm() > physics.max_speed) {
            target.linear_velocity = target.linear_velocity.normalized() * physics.max_speed;
        }

        if (target.angular_velocity > physics.max_angular_velocity) {
            target.angular_velocity = physics.max_angular_velocity;
        }
        if (target.angular_velocity < -physics.max_angular_velocity) {
            target.angular_velocity = -physics.max_angular_velocity;
        }
    }


    State MultiAgentCityEnv::reset(std::optional<unsigned int> seed) {
        if (seed.has_value()) {
            this->seed(seed.value());
        }
        drone.position.vector.setZero();
        drone.position.yaw = 0.0f;
        drone.linear_velocity.setZero();
        drone.angular_velocity = 0.0f;
        
    

        if (this->randomize_target_on_reset) {
            
            Eigen::Vector2i random_grid_pos;
            std::uniform_int_distribution<int> x_dist(0, obstacle_map[0].size() - 1);
            std::uniform_int_distribution<int> y_dist(0, obstacle_map.size() - 1);

            do {
                random_grid_pos.y() = y_dist(random_generator);
                random_grid_pos.x() = x_dist(random_generator);
            } while (obstacle_map[random_grid_pos.y()][random_grid_pos.x()]);


            target.position.vector = mapToWorld(random_grid_pos);

        } else {
            // Use the fixed starting position provided during construction
            target.position.vector = this->initial_target_position;
        }
        target.position.yaw = 0.0f;
        target.linear_velocity.setZero();  
        target.angular_velocity = 0.0f;

        time_elapsed = 0.0f;
        target.path.clear();
        target.current_path_index = 0;

        return get_state();
    }





































}