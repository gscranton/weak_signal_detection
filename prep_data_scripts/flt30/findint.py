"""
exec(open("findint.py").read())
"""

import numpy as np
import matplotlib.pyplot as plt

def find_intersection(trajectories):
    """
    Finds the point that minimizes the distance to a set of 2D trajectories.
    
    Args:
        trajectories: List of arrays/lists, each of shape (N, 2) containing [x, y]
        
    Returns:
        (x, y): The estimated intersection point.
    """
    A = []
    B = []

    for traj in trajectories:
        traj = np.array(traj)
        # 1. Find the best-fit line for this trajectory using PCA
        # This gives us a point on the line (the mean) and the direction vector
        centroid = np.mean(traj, axis=0)
        u, s, vh = np.linalg.svd(traj - centroid)
        direction = vh[0]  # First principal component (direction of the line)
        
        # 2. Convert line to the form: ax + by = c
        # The normal vector (a, b) is perpendicular to the direction vector
        normal = np.array([-direction[1], direction[0]])
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        a, b = normal
        c = np.dot(normal, centroid)
        
        A.append([a, b])
        B.append(c)

    # 3. Solve the overdetermined system Ax = B using least squares
    A = np.array(A)
    B = np.array(B)
    intersection, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    
    return intersection

"""
# --- Example Usage with Synthetic Data ---

# Generate 5 trajectories intersecting roughly at (50, 50)
np.random.seed(42)
true_intersect = np.array([50, 50])
trajectories = []

for i in range(5):
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.linspace(-30, -5, 20) # Points approaching the center
    # Create linear path with noise
    x = true_intersect[0] + dist * np.cos(angle) + np.random.normal(0, 1, 20)
    y = true_intersect[1] + dist * np.sin(angle) + np.random.normal(0, 1, 20)
    trajectories.append(np.column_stack((x, y)))

# Find the intersection
est_x, est_y = find_intersection(trajectories)

print(f"True Intersection: {true_intersect}")
print(f"Estimated Intersection: ({est_x:.2f}, {est_y:.2f})")

# Visualization
plt.figure(figsize=(8, 6))
for i, traj in enumerate(trajectories):
    plt.scatter(traj[:, 0], traj[:, 1], s=10, label=f'Flight {i+1}')
    # Extend the lines to show the intersection
    z = np.polyfit(traj[:, 0], traj[:, 1], 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(traj[:, 0]), est_x, 100)
    plt.plot(x_range, p(x_range), '--', alpha=0.5)

plt.plot(est_x, est_y, 'ro', markersize=10, label='Estimated Intersection')
plt.legend()
plt.grid(True)
plt.title("Finding Flight Path Intersection Point")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show(block=False)
"""
