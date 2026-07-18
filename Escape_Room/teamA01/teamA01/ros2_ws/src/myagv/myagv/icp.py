import numpy as np
from scipy.spatial import cKDTree

def laser_scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """
    Μετατρέπει τις αποστάσεις του Lidar (Polar) σε 2D σημεία (Cartesian X, Y)
    στο τοπικό σύστημα συντεταγμένων του ρομπότ.
    """
    points = []
    for i, r in enumerate(ranges):
        if range_min < r < range_max and not np.isinf(r) and not np.isnan(r):
            angle = angle_min + (i * angle_increment)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
    return np.array(points)

def transform_points(points, x, y, theta):
    """Μετασχηματίζει ένα σύνολο σημείων κατά X, Y και Θ."""
    if len(points) == 0:
        return points
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], 
                  [sin_t,  cos_t]])
    # Περιστροφή και μετατόπιση
    return np.dot(points, R.T) + np.array([x, y])

def icp_2d(source, target, initial_guess=(0.0, 0.0, 0.0), max_iterations=20, tolerance=0.001):
    """
    Custom 2D ICP αλγόριθμος αναβαθμισμένος με KD-Tree για ταχύτητα.
    """
    if len(source) == 0 or len(target) == 0:
        return 0.0, 0.0, 0.0

    T_cumulative_x, T_cumulative_y, T_cumulative_theta = initial_guess
    current_source = transform_points(source.copy(), T_cumulative_x, T_cumulative_y, T_cumulative_theta)

    # Φτιάχνουμε το KD-Tree μια φορά για το target scan. 
    target_tree = cKDTree(target)

    # Μέγιστη επιτρεπτή απόσταση για να θεωρηθεί "ταίρι" (0.3 μέτρα)
    MAX_MATCH_DISTANCE = 0.3  

    for _ in range(max_iterations):
        # Εύρεση Κοντινότερων Σημείων με KD-Tree
        distances, matched_indices = target_tree.query(current_source)

        # Φιλτράρισμα: Κρατάμε μόνο τα σημεία που είναι κοντά
        valid_matches = distances < MAX_MATCH_DISTANCE
        
        # Αν χάσαμε τα περισσότερα σημεία, σταματάμε
        if np.sum(valid_matches) < 10: 
            break

        valid_source = current_source[valid_matches]
        valid_target = target[matched_indices[valid_matches]]

        # Υπολογισμός Κέντρων Βάρους (Centroids)
        centroid_source = np.mean(valid_source, axis=0)
        centroid_target = np.mean(valid_target, axis=0)

        # Centering 
        s_centered = valid_source - centroid_source
        t_centered = valid_target - centroid_target

        # SVD (Kabsch)
        H = np.dot(s_centered.T, t_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_target - np.dot(R, centroid_source)

        dx = t[0]
        dy = t[1]
        dtheta = np.arctan2(R[1, 0], R[0, 0])

        # Ενημέρωση σημείων
        current_source = transform_points(current_source, dx, dy, dtheta)
        
        T_cumulative_x += dx
        T_cumulative_y += dy
        T_cumulative_theta += dtheta

        # Κριτήριο σύγκλισης 
        if np.sqrt(dx**2 + dy**2) + abs(dtheta) < tolerance:
            break

    T_cumulative_theta = np.arctan2(np.sin(T_cumulative_theta), np.cos(T_cumulative_theta))
    return T_cumulative_x, T_cumulative_y, T_cumulative_theta
