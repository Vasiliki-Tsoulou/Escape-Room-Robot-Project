import numpy as np
import math
from skimage.draw import line

def update_occupancy_grid(grid, map_resolution, origin_x, origin_y, 
                          robot_x, robot_y, robot_theta, 
                          laser_ranges, angle_min, angle_increment, 
                          range_min, range_max, unlocked_cells_set):
    """
    Ενημερώνει τον 2D χάρτη (Numpy Array) χρησιμοποιώντας Raycasting (Bresenham).
    """
    
    # Βρίσκουμε σε ποιο κελί του χάρτη βρίσκεται το ρομπότ
    robot_c = int((robot_x - origin_x) / map_resolution)
    robot_r = int((robot_y - origin_y) / map_resolution)
    
    rows, cols = grid.shape
    
    # Μέγιστη απόσταση που "καθαρίζει" το ρομπότ όταν η ακτίνα χάνεται στο κενό (σε μέτρα)
    MAX_CLEARING_DIST = 5.0 
    MAX_RELIABLE_DIST = 5.0

    # Σαρώνουμε κάθε ακτίνα του Lidar
    for i, r in enumerate(laser_ranges):
        hit_obstacle = True  # Υποθέτουμε αρχικά ότι χτυπάμε τοίχο
        
        # Έλεγχος για free space (φιλτραρισμένα 0.0, άπειρο, ή εκτός εμβέλειας)
        if math.isinf(r) or r > MAX_RELIABLE_DIST:
            r = MAX_CLEARING_DIST
            hit_obstacle = False  # Κοιτάμε στο κενό, δεν χτυπήσαμε εμπόδιο
        
        elif r < range_min or r == 0 or math.isnan(r):
            continue  # Hardware Error του Lidar, το αγνοούμε
            
        # Υπολογισμός της παγκόσμιας γωνίας της ακτίνας
        ray_angle = robot_theta + angle_min + (i * angle_increment)
        
        # Πού φτάνει η ακτίνα στον πραγματικό κόσμο
        hit_x = robot_x + r * math.cos(ray_angle)
        hit_y = robot_y + r * math.sin(ray_angle)
        
        # Μετατροπή του σημείου σε κελί του πλέγματος
        hit_c = int((hit_x - origin_x) / map_resolution)
        hit_r = int((hit_y - origin_y) / map_resolution)
        
        # Κρατάμε την ακτίνα εντος στα όρια του χάρτη (Clamping) για να μην σκάσει η Bresenham
        hit_r = max(0, min(rows - 1, hit_r))
        hit_c = max(0, min(cols - 1, hit_c))
            
        # Bresenham algorithm
        rr, cc = line(robot_r, robot_c, hit_r, hit_c)

        valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        rr, cc = rr[valid], cc[valid]
        
        if len(rr) == 0:
            continue
        
        # όλα τα ενδιάμεσα κελιά είναι πάντα ελεύθερα (ενεργή αφαίρεση ghost του χάρτη)
        grid[rr[:-1], cc[:-1]] = 0
        
        # τελευταίο κελί της ακτίνας
        if hit_obstacle:
            # Χτύπησε τοίχο. Ελέγχουμε αν είναι ξεκλειδωμένη πόρτα.
            if (hit_c, hit_r) in unlocked_cells_set:
                grid[hit_r, hit_c] = 0
            else:
                grid[hit_r, hit_c] = 100
        else:
            # Ήταν ανοιχτός διάδρομος/κενό. Το τελευταίο κελί είναι απλά αέρας.
            grid[hit_r, hit_c] = 0

    return grid
