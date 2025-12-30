"""
Example experiment using the polyhedra library.

Run: python experiment.py
"""

from polyhedra import Octahedron, Icosahedron, Tetrahedron, count_union_faces, PHI
import numpy as np

print("=" * 60)
print("Polyhedra Face Counting Experiments")
print("Using trimesh boolean union for accurate counts")
print("=" * 60)

# ---------------------------------------------------------
# Experiment 1: Two octahedra at various angles
# ---------------------------------------------------------
print("\n--- Experiment 1: Two Octahedra ---")
print("Rotation around Y axis:")
print("Angle | Faces")
print("-" * 15)
for angle in range(0, 91, 15):
    oct1 = Octahedron()
    oct2 = Octahedron().rotate([0, 1, 0], angle)
    count = count_union_faces([oct1, oct2])
    print(f"{angle:5}° | {count}")

# ---------------------------------------------------------
# Experiment 2: Two tetrahedra
# ---------------------------------------------------------
print("\n--- Experiment 2: Two Tetrahedra ---")
print("Rotation around (1,1,1) axis:")
for angle in [0, 60, 120, 180]:
    tet1 = Tetrahedron()
    tet2 = Tetrahedron().rotate([1, 1, 1], angle)
    count = count_union_faces([tet1, tet2])
    print(f"{angle:5}° | {count} faces")

# ---------------------------------------------------------
# Experiment 3: 5 Octahedra (like the 5-JB context)
# ---------------------------------------------------------
print("\n--- Experiment 3: 5 Octahedra (5-fold symmetry) ---")
axis = [0, 1, PHI]  # 5-fold axis of icosahedral symmetry

octs = [Octahedron().rotate(axis, i * 72) for i in range(5)]
count = count_union_faces(octs)
print(f"5 octahedra around 5-fold axis: {count} faces")

# ---------------------------------------------------------
# Experiment 4: Single shapes
# ---------------------------------------------------------
print("\n--- Experiment 4: Single Platonic Solids ---")
print(f"Tetrahedron: {count_union_faces([Tetrahedron()])} faces")
print(f"Octahedron: {count_union_faces([Octahedron()])} faces")
print(f"Icosahedron: {count_union_faces([Icosahedron()])} faces")

# ---------------------------------------------------------
# Experiment 5: Two icosahedra
# ---------------------------------------------------------
print("\n--- Experiment 5: Two Icosahedra ---")
for angle in [0, 36, 72]:
    ico1 = Icosahedron()
    ico2 = Icosahedron().rotate([0, 1, 0], angle)
    count = count_union_faces([ico1, ico2])
    print(f"Rotation {angle}° | {count} faces")
