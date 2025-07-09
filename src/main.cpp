// src/main.cpp

#include "CityEnv.hpp"
#include <iostream> // For basic output
#include <string>
#include <fstream>
#include <sstream>


std::vector<std::vector<bool>> load_obstacle_map_from_csv(const std::string& filename);

int main() {

    std::cout << "Starting CityEnv simulation..." << std::endl;

    // Load the obstacle map from a CSV file

    std::string filename = "/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/src/obstacles.csv";
    std::vector<std::vector<bool>> obstacle_map = load_obstacle_map_from_csv(filename);
    std::cout << "Obstacle map loaded with " << obstacle_map.size() << " rows." << std::endl;
   
    city_env::Drone drone; // Create a drone object
    drone.id = 0; // Set a default ID for the drone
    drone.position.vector.setZero(); // Initialize position to zero
    drone.position.yaw = 0.0f; // Initialize yaw to zero
    drone.linear_velocity.setZero(); // Initialize linear velocity to zero
    drone.angular_velocity = 0.0f; // Initialize angular velocity to zero
    city_env::Target target; // Create a target object
    target.position.vector.setZero(); // Initialize target position to zero
    target.position.yaw = 0.0f; // Initialize target yaw to zero        
    city_env::CityEnv city_env(
        obstacle_map, // Obstacle map loaded from CSV
        1000.0f, // World width
        1000.0f, // World height
        1.0f / 60.0f, // Time step
        90.0f, // Field of view angle
        100.0f, // Field of view distance
        drone, // Single drone object
        target // Single target object
        );



    return 0; // Indicate successful execution
}



std::vector<std::vector<bool>> load_obstacle_map_from_csv(const std::string& filename) {
    std::vector<std::vector<bool>> obstacle_map;
    std::ifstream file(filename); // Open the file for reading

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return obstacle_map; // Return an empty map on failure
    }

    std::string line;
    // Read the file line by line
    while (std::getline(file, line)) {
        std::vector<bool> row;
        std::stringstream ss(line); // Create a string stream from the line
        std::string cell;

        // Split the line by commas
        while (std::getline(ss, cell, ',')) {
            try {
                // Convert the cell string to an integer, then to a boolean
                row.push_back(std::stoi(cell) == 1);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Invalid non-integer value '" << cell << "' in CSV. Treating as no obstacle." << std::endl;
                row.push_back(false);
            } catch (const std::out_of_range& e) {
                 std::cerr << "Warning: Value '" << cell << "' is out of range. Treating as no obstacle." << std::endl;
                 row.push_back(false);
            }
        }
        obstacle_map.push_back(row);
    }

    file.close();
    std::cout << "Successfully loaded obstacle map from " << filename << std::endl;
    return obstacle_map;
}
