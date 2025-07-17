// src/main.cpp

#include "CityEnv.hpp"
#include <iostream> // For basic output
#include <string>
#include <fstream>
#include <sstream>


std::vector<std::vector<bool>> load_obstacle_map_from_csv(const std::string& filename);

int main() {



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
