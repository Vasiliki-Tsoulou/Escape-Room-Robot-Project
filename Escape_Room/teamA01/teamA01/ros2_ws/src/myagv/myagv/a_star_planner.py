import heapq
import math
import numpy as np

class Node:
    """
    A node class for A* Pathfinding.
    Stores grid coordinates, cost values, and the parent node for path reconstruction.
    """
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        
        self.g = 0.0  # Cost from start to this node
        self.h = 0.0  # Estimated heuristic cost to goal
        self.f = 0.0  # Total cost (g + h)


    # Define less-than for the priority queue to sort by lowest f-cost
    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def heuristic(current, goal, weight=1.5):
    """
    Calculates the Octile distance heuristic - Weighted
    """
    dx = abs(current.x - goal.x)
    dy = abs(current.y - goal.y)
    
    # Octile distance :
    h = (dx + dy) - 0.586 * min(dx, dy)
    
    return h * weight

def get_neighbors(node, grid):
    neighbors = []
    rows, cols = grid.shape
    
    # 4-way orthogonal moves
    ortho = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in ortho:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
            neighbors.append((nx, ny, 1.0))
            
    # 4-way diagonal moves
    diags = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for dx, dy in diags:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
            
            # Check the two adjacent orthogonal cells to prevent clipping the wall
            if grid[node.x + dx][node.y] == 0 and grid[node.x][node.y + dy] == 0:
                neighbors.append((nx, ny, 1.414))
                
    return neighbors

def a_star_search(grid, start_coords, goal_coords, weight=1.5):
    """
    Executes the A* pathfinding algorithm on a 2D grid.
    Returns a list of (x, y) tuples representing the path from start to goal.
    """
    start_node = Node(start_coords[0], start_coords[1])
    goal_node = Node(goal_coords[0], goal_coords[1])
    
    # Priority queue for nodes to be evaluated
    open_list = []
    heapq.heappush(open_list, start_node)
    
    rows, cols = grid.shape
    g_costs = np.full((rows, cols), np.inf)
    g_costs[start_node.x, start_node.y] = 0.0
    
    while open_list:
        current_node = heapq.heappop(open_list)
        
        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1] # Return reversed path
            
        # Generate and evaluate neighbors
        for nx, ny, move_cost in get_neighbors(current_node, grid):
            neighbor = Node(nx, ny, parent=current_node)
            tentative_g = current_node.g + move_cost
            
            # process if better
            if tentative_g < g_costs[nx, ny]:
                g_costs[nx, ny] = tentative_g
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, goal_node, weight)
                neighbor.f = neighbor.g + neighbor.h
                
                heapq.heappush(open_list, neighbor)
                
    # Return empty list if no path is found
    return []


def main(args=None):
    #testing function

    # 0 = Free Space, 1 = Wall/Locked Door
    test_grid = [
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 5)

    test_grid_np = np.array(test_grid)
    
    # Run the algorithm
    calculated_path = a_star_search(test_grid_np, start, goal, weight=1.5)
    
    print("Calculated Path Coordinates:")
    print(calculated_path)
    
    # Visualizing the path on the grid in the terminal
    print("\nVisualized Grid (S=Start, G=Goal, *=Path, 1=Wall):")
    for r in range(len(test_grid)):
        row_str = ""
        for c in range(len(test_grid[0])):
            if (r, c) == start:
                row_str += " S "
            elif (r, c) == goal:
                row_str += " G "
            elif (r, c) in calculated_path:
                row_str += " * "
            elif test_grid[r][c] == 1:
                row_str += " 1 "
            else:
                row_str += " . "
        print(row_str)

if __name__ == '__main__':
    main()