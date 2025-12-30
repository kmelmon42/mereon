"""
6 Icosahedra Face Counter

Experiment with different constructions to find one that gives ~300 faces.

Usage: python count_6_icosahedra.py
"""

import trimesh
import numpy as np
from polyhedra import Icosahedron, PHI, rotation_matrix

def count_faces(icosahedra, area_threshold=0.0):
    """
    Count faces on the boolean union of icosahedra.
    
    area_threshold: Only count faces with area >= this value
                    Set to 0 to count all faces
    """
    meshes = []
    for ico in icosahedra:
        verts = ico.get_vertices()
        faces = np.array(ico.get_faces())
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        meshes.append(mesh)
    
    if len(meshes) == 1:
        return len(meshes[0].faces)
    
    union = trimesh.boolean.union(meshes)
    
    if area_threshold > 0:
        areas = union.area_faces
        return (areas >= area_threshold).sum()
    
    return len(union.faces)

# =============================================================================
# CONSTRUCTION PARAMETERS - MODIFY THESE
# =============================================================================

# Alignment angle to put vertex on Y axis
align_angle = np.arctan2(1, PHI) * 180 / np.pi  # ~31.72°

# The magic angle for bisector rotation
atan_half = np.arctan(0.5) * 180 / np.pi  # ~26.565°

# 5 bisector axes (perpendicular to triangle bisector planes)
BISECTOR_AXES = [
    [-0.9511, 0, -0.3090],   # Triangle 0
    [0.0000, 0, -1.0000],    # Triangle 1
    [0.9511, 0, -0.3090],    # Triangle 2
    [0.5878, 0, 0.8090],     # Triangle 3
    [-0.5878, 0, 0.8090],    # Triangle 4
]

# =============================================================================
# BUILD THE 6 ICOSAHEDRA
# =============================================================================

def build_6_icosahedra_v1():
    """Original construction from visualization"""
    icosahedra = []
    
    for i in range(6):
        ico = Icosahedron()
        
        # Step 1: Align vertex to Y axis
        ico.rotate([0, 0, 1], align_angle)
        
        # Step 2: Flip 180° around Y (for icosahedra 2-6)
        if i > 0:
            ico.rotate([0, 1, 0], 180)
        
        # Step 3: Rotate around bisector axis (for icosahedra 2-6)
        if i > 0:
            bisector_idx = i - 1
            ico.rotate(BISECTOR_AXES[bisector_idx], atan_half)
        
        icosahedra.append(ico)
    
    return icosahedra

def build_6_icosahedra_v2():
    """Alternative: no flip, just bisector rotations"""
    icosahedra = []
    
    for i in range(6):
        ico = Icosahedron()
        ico.rotate([0, 0, 1], align_angle)
        
        if i > 0:
            bisector_idx = i - 1
            ico.rotate(BISECTOR_AXES[bisector_idx], atan_half)
        
        icosahedra.append(ico)
    
    return icosahedra

def build_6_icosahedra_v3():
    """Alternative: 60° rotations around Y"""
    icosahedra = []
    
    for i in range(6):
        ico = Icosahedron()
        ico.rotate([0, 1, 0], i * 60)
        icosahedra.append(ico)
    
    return icosahedra

# =============================================================================
# TEST DIFFERENT CONSTRUCTIONS
# =============================================================================

print("=" * 60)
print("6 Icosahedra Face Counter")
print("=" * 60)

print("\nReference:")
print(f"  Single icosahedron: {count_faces([Icosahedron()])} faces")
print(f"  6 × 20 = 120 original faces")

print("\n--- Construction V1 (original, with flip) ---")
icos = build_6_icosahedra_v1()
print(f"  Total faces (union): {count_faces(icos)}")
print(f"  Faces with area >= 0.01: {count_faces(icos, 0.01)}")
print(f"  Faces with area >= 0.05: {count_faces(icos, 0.05)}")

print("\n--- Construction V2 (no flip) ---")
icos = build_6_icosahedra_v2()
print(f"  Total faces (union): {count_faces(icos)}")
print(f"  Faces with area >= 0.01: {count_faces(icos, 0.01)}")

print("\n--- Construction V3 (simple 60° rotations) ---")
icos = build_6_icosahedra_v3()
print(f"  Total faces (union): {count_faces(icos)}")
print(f"  Faces with area >= 0.01: {count_faces(icos, 0.01)}")

# =============================================================================
# SEARCH FOR ANGLE THAT GIVES ~300
# =============================================================================

print("\n--- Searching for angle that gives ~300 faces ---")
print("(No flip, varying bisector angle)")

for angle in range(0, 91, 5):
    icosahedra = []
    for i in range(6):
        ico = Icosahedron()
        ico.rotate([0, 0, 1], align_angle)
        if i > 0:
            bisector_idx = i - 1
            ico.rotate(BISECTOR_AXES[bisector_idx], angle)
        icosahedra.append(ico)
    
    total = count_faces(icosahedra)
    significant = count_faces(icosahedra, 0.01)
    
    if 280 <= significant <= 320:
        print(f"  *** Angle {angle}°: {total} total, {significant} significant ***")
    elif angle % 15 == 0:
        print(f"  Angle {angle}°: {total} total, {significant} significant")
