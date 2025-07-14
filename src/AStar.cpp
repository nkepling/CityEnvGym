#include "AStar.hpp"

#include <queue>       // For std::priority_queue
#include <unordered_set> // For the "closed list"
#include <vector>
#include <cmath>       // For std::abs
#include <algorithm>   // For std::reverse
#include <iostream>    // For debugging output

namespace AStar {

// Represents a node in the search grid
struct Node {
    Vec2i position;
    int g_cost = 0; // Cost from the start node
    int h_cost = 0; // Heuristic cost to the goal node
    Node* parent = nullptr;

    int get_f_cost() const {
        return g_cost + h_cost;
    }

    // Needed for storing nodes in a set
    bool operator==(const Node& other) const {
        return position.x() == other.position.x() && position.y() == other.position.y();
    }
};

// Custom comparator for the priority queue to sort by F-cost
struct CompareNode {
    bool operator()(const Node* a, const Node* b) const {
        return a->get_f_cost() > b->get_f_cost();
    }
};

// Custom hash function for Vec2i to use it in an unordered_set
struct Vec2iHash {
    std::size_t operator()(const Vec2i& v) const {
        // A simple hash combination
        return std::hash<int>()(v.x()) ^ (std::hash<int>()(v.y()) << 1);
    }
};

// Heuristic function (Manhattan distance)
int heuristic(const Vec2i& a, const Vec2i& b) {
    return std::abs(a.x() - b.x()) + std::abs(a.y() - b.y());
}

// Reconstructs the final path by tracing parent pointers from the goal
std::vector<Vec2i> reconstructPath(Node* goal_node) {
    std::vector<Vec2i> path;
    Node* current = goal_node;
    while (current != nullptr) {
        path.push_back(current->position);
        current = current->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// Main A* function implementation
// Main A* function implementation
std::vector<Vec2i> findPath(
    const Vec2i& start,
    const Vec2i& goal,
    const std::vector<std::vector<bool>>& obstacle_map,
    bool allow_diagonal
) {
    if (obstacle_map.empty() || obstacle_map[0].empty()) {
        return {};
    }
    const int world_height = obstacle_map.size();
    const int world_width = obstacle_map[0].size();

    std::vector<Node*> all_nodes;
    auto create_node = [&](const Vec2i& pos, int g, int h, Node* p) {
        Node* n = new Node{pos, g, h, p};
        all_nodes.push_back(n);
        return n;
    };

    std::priority_queue<Node*, std::vector<Node*>, CompareNode> open_list;
    std::unordered_set<Vec2i, Vec2iHash> closed_list;

    Node* start_node = create_node(start, 0, heuristic(start, goal), nullptr);
    open_list.push(start_node);

    std::vector<Vec2i> directions = {
        {0, 1}, {1, 0}, {0, -1}, {-1, 0},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    if (!allow_diagonal) {
        directions.resize(4);
    }

    while (!open_list.empty()) {
        Node* current_node = open_list.top();
        open_list.pop();

        // If this position is in the closed list, it means we have already
        // found a better path to it. So, we skip this one.
        if (closed_list.count(current_node->position)) {
            continue;
        }

        // Add the current position to the closed list because we are now
        // processing the best path to this point.
        closed_list.insert(current_node->position);

        // Goal reached
        if (current_node->position == goal) {
            std::vector<Vec2i> path = reconstructPath(current_node);
            for (Node* n : all_nodes) delete n;
            return path;
        }

        // Check neighbors
        for (const auto& dir : directions) {
            Vec2i neighbor_pos = current_node->position + dir;

            if (neighbor_pos.x() < 0 || neighbor_pos.x() >= world_width ||
                neighbor_pos.y() < 0 || neighbor_pos.y() >= world_height ||
                obstacle_map[neighbor_pos.y()][neighbor_pos.x()]) {
                continue;
            }
            
            // The check for the closed list is removed from here.
            // We create a new node and add it to the open list.
            int tentative_g_cost = current_node->g_cost + ((dir.x() == 0 || dir.y() == 0) ? 10 : 14);
            Node* neighbor_node = create_node(
                neighbor_pos,
                tentative_g_cost,
                heuristic(neighbor_pos, goal),
                current_node
            );
            open_list.push(neighbor_node);
        }
    }

    for (Node* n : all_nodes) delete n;
    std::cout << "No path found from " << start.transpose() << " to " << goal.transpose() << std::endl;
    return {};
}
} // namespace AStar