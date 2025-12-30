"""
polyhedra.py - Core library for polyhedra compound analysis

Properly handles:
- Face subdivision when cut by other polyhedra
- Coplanar face merging/intersection
- Visibility testing for each sub-region

Usage:
    from polyhedra import Octahedron, Icosahedron, count_visible_faces
    
    oct1 = Octahedron()
    oct2 = Octahedron().rotate([0,1,0], 45)
    print(count_visible_faces([oct1, oct2]))
"""

import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import split, unary_union
from shapely import make_valid
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict

# Constants
PHI = (1 + np.sqrt(5)) / 2
TOL = 1e-9

# =============================================================================
# Core Geometry Functions
# =============================================================================

def normalize(v):
    """Normalize a vector"""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < TOL:
        return v
    return v / n

def rotation_matrix(axis, angle_degrees):
    """Create rotation matrix for rotation around axis by angle (degrees)"""
    axis = normalize(np.array(axis, dtype=float))
    angle = angle_degrees * np.pi / 180
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])

def triangle_normal(p0, p1, p2):
    """Compute normal of triangle"""
    v1 = np.array(p1) - np.array(p0)
    v2 = np.array(p2) - np.array(p0)
    n = np.cross(v1, v2)
    return normalize(n)

def polygon_normal(vertices):
    """Compute normal of a polygon (uses first 3 vertices)"""
    if len(vertices) < 3:
        return np.array([0, 0, 1])
    return triangle_normal(vertices[0], vertices[1], vertices[2])

def ray_triangle_intersection(ray_origin, ray_dir, v0, v1, v2):
    """Möller–Trumbore ray-triangle intersection."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < TOL:
        return None, False
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < -TOL or u > 1.0 + TOL:
        return None, False
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    
    if v < -TOL or u + v > 1.0 + TOL:
        return None, False
    
    t = f * np.dot(edge2, q)
    
    if t > TOL:
        return t, True
    
    return None, False

def triangulate_polygon(vertices):
    """Fan triangulation of a convex polygon"""
    if len(vertices) < 3:
        return []
    triangles = []
    for i in range(1, len(vertices) - 1):
        triangles.append([vertices[0], vertices[i], vertices[i+1]])
    return triangles

def point_inside_polyhedron(point, vertices, faces):
    """Test if point is inside polyhedron using ray casting."""
    ray_dir = np.array([1.0, 0.001, 0.002])  # Slightly off-axis to avoid edge cases
    ray_dir = normalize(ray_dir)
    count = 0
    
    for face in faces:
        face_verts = [vertices[i] for i in face]
        triangles = triangulate_polygon(face_verts)
        
        for tri in triangles:
            t, hit = ray_triangle_intersection(point, ray_dir, 
                                               np.array(tri[0]), 
                                               np.array(tri[1]), 
                                               np.array(tri[2]))
            if hit:
                count += 1
    
    return count % 2 == 1

# =============================================================================
# 3D to 2D Projection for Polygon Operations
# =============================================================================

def get_plane_basis(normal):
    """Get two orthonormal vectors in the plane with given normal"""
    normal = normalize(normal)
    
    # Find a vector not parallel to normal
    if abs(normal[0]) < 0.9:
        helper = np.array([1, 0, 0])
    else:
        helper = np.array([0, 1, 0])
    
    u = normalize(np.cross(normal, helper))
    v = normalize(np.cross(normal, u))
    
    return u, v

def project_to_2d(points_3d, origin, u, v):
    """Project 3D points onto 2D plane defined by origin, u, v"""
    points_2d = []
    for p in points_3d:
        rel = np.array(p) - origin
        x = np.dot(rel, u)
        y = np.dot(rel, v)
        points_2d.append((x, y))
    return points_2d

def unproject_to_3d(points_2d, origin, u, v):
    """Convert 2D points back to 3D"""
    points_3d = []
    for x, y in points_2d:
        p = origin + x * u + y * v
        points_3d.append(p)
    return points_3d

# =============================================================================
# Edge-Plane Intersection
# =============================================================================

def edge_plane_intersection(p1, p2, plane_normal, plane_d):
    """
    Find intersection of edge (p1, p2) with plane (normal · x = d)
    Returns intersection point or None
    """
    p1, p2 = np.array(p1), np.array(p2)
    d1 = np.dot(plane_normal, p1) - plane_d
    d2 = np.dot(plane_normal, p2) - plane_d
    
    # Both on same side
    if d1 * d2 > TOL:
        return None
    
    # Edge lies in plane
    if abs(d1) < TOL and abs(d2) < TOL:
        return None
    
    # One endpoint on plane
    if abs(d1) < TOL:
        return p1
    if abs(d2) < TOL:
        return p2
    
    # Proper intersection
    t = d1 / (d1 - d2)
    return p1 + t * (p2 - p1)

# =============================================================================
# Polyhedron Class
# =============================================================================

class Polyhedron:
    """Base class for polyhedra with transformation support"""
    
    def __init__(self, vertices, faces):
        self.base_vertices = np.array(vertices, dtype=float)
        self.faces = [list(f) for f in faces]
        self.transform = np.eye(3)
        self.translation = np.zeros(3)
        self.scale_factor = 1.0
    
    def rotate(self, axis, angle_degrees):
        """Rotate around axis by angle (degrees). Returns self for chaining."""
        rot = rotation_matrix(axis, angle_degrees)
        self.transform = rot @ self.transform
        return self
    
    def translate(self, offset):
        """Translate by offset vector. Returns self for chaining."""
        self.translation = self.translation + np.array(offset)
        return self
    
    def scale(self, factor):
        """Scale uniformly. Returns self for chaining."""
        self.scale_factor *= factor
        return self
    
    def get_vertices(self):
        """Get transformed vertices"""
        v = self.base_vertices * self.scale_factor
        v = v @ self.transform.T
        v = v + self.translation
        return v
    
    def get_faces(self):
        """Get face index lists"""
        return self.faces
    
    def get_face_vertices(self, face_idx):
        """Get the actual vertex positions for a face"""
        verts = self.get_vertices()
        return [verts[i] for i in self.faces[face_idx]]
    
    def copy(self):
        """Create a copy of this polyhedron with same transforms"""
        p = Polyhedron(self.base_vertices.copy(), [f.copy() for f in self.faces])
        p.transform = self.transform.copy()
        p.translation = self.translation.copy()
        p.scale_factor = self.scale_factor
        return p

# =============================================================================
# Platonic Solids
# =============================================================================

class Tetrahedron(Polyhedron):
    def __init__(self):
        vertices = [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ]
        super().__init__(vertices, faces)

class Octahedron(Polyhedron):
    def __init__(self):
        vertices = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]
        ]
        faces = [
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
            [1, 4, 2],
            [1, 3, 4],
            [1, 5, 3],
            [1, 2, 5]
        ]
        super().__init__(vertices, faces)

class Cube(Polyhedron):
    def __init__(self):
        vertices = [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1]
        ]
        faces = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5]
        ]
        super().__init__(vertices, faces)

class Icosahedron(Polyhedron):
    def __init__(self):
        vertices = [
            [0, 1, PHI],
            [0, 1, -PHI],
            [0, -1, PHI],
            [0, -1, -PHI],
            [1, PHI, 0],
            [1, -PHI, 0],
            [-1, PHI, 0],
            [-1, -PHI, 0],
            [PHI, 0, 1],
            [PHI, 0, -1],
            [-PHI, 0, 1],
            [-PHI, 0, -1]
        ]
        faces = [
            [0, 2, 8],
            [0, 8, 4],
            [0, 4, 6],
            [0, 6, 10],
            [0, 10, 2],
            [1, 4, 9],
            [1, 9, 3],
            [1, 3, 11],
            [1, 11, 6],
            [1, 6, 4],
            [2, 7, 5],
            [2, 5, 8],
            [2, 10, 7],
            [3, 5, 7],
            [3, 7, 11],
            [3, 9, 5],
            [4, 8, 9],
            [5, 9, 8],
            [6, 11, 10],
            [7, 10, 11]
        ]
        super().__init__(vertices, faces)

class Dodecahedron(Polyhedron):
    def __init__(self):
        phi = PHI
        iphi = 1/PHI
        vertices = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, iphi], [0, phi, -iphi], [0, -phi, iphi], [0, -phi, -iphi],
            [iphi, 0, phi], [iphi, 0, -phi], [-iphi, 0, phi], [-iphi, 0, -phi],
            [phi, iphi, 0], [phi, -iphi, 0], [-phi, iphi, 0], [-phi, -iphi, 0]
        ]
        faces = [
            [0, 8, 4, 14, 12],
            [0, 12, 2, 17, 16],
            [0, 16, 1, 9, 8],
            [1, 16, 17, 3, 13],
            [1, 13, 15, 5, 9],
            [2, 12, 14, 6, 10],
            [2, 10, 11, 3, 17],
            [3, 11, 7, 15, 13],
            [4, 8, 9, 5, 18],
            [4, 18, 19, 6, 14],
            [5, 15, 7, 19, 18],
            [6, 19, 7, 11, 10]
        ]
        super().__init__(vertices, faces)

# =============================================================================
# Face Counting - Main Algorithm
# =============================================================================

def get_plane_key(normal, d, tol=1e-6):
    """Create a hashable key for a plane (handles floating point)"""
    # Ensure consistent normal direction
    if normal[0] < -tol or (abs(normal[0]) < tol and normal[1] < -tol) or \
       (abs(normal[0]) < tol and abs(normal[1]) < tol and normal[2] < 0):
        normal = -normal
        d = -d
    
    # Round for hashing
    n = tuple(round(x, 5) for x in normal)
    d = round(d, 5)
    return (n, d)

def collect_all_faces(polyhedra):
    """
    Collect all faces from all polyhedra with their plane info.
    Returns list of dicts with face data.
    """
    all_faces = []
    
    for poly_idx, poly in enumerate(polyhedra):
        verts = poly.get_vertices()
        
        for face_idx, face in enumerate(poly.get_faces()):
            face_verts = [verts[i] for i in face]
            normal = polygon_normal(face_verts)
            d = np.dot(normal, face_verts[0])
            plane_key = get_plane_key(normal, d)
            
            all_faces.append({
                'poly_idx': poly_idx,
                'face_idx': face_idx,
                'vertices_3d': face_verts,
                'normal': normal,
                'd': d,
                'plane_key': plane_key
            })
    
    return all_faces

def get_cutting_segments(face_data, all_faces, polyhedra):
    """
    Find all line segments that cut through this face from other polyhedra.
    """
    face_verts = face_data['vertices_3d']
    normal = face_data['normal']
    d = face_data['d']
    poly_idx = face_data['poly_idx']
    
    # Get 2D basis for this face's plane
    origin = np.array(face_verts[0])
    u, v = get_plane_basis(normal)
    
    # Project face to 2D
    face_2d = project_to_2d(face_verts, origin, u, v)
    try:
        face_polygon = Polygon(face_2d)
        if not face_polygon.is_valid:
            face_polygon = make_valid(face_polygon)
    except:
        return []
    
    cutting_lines = []
    
    # Check all edges from other polyhedra
    for other_poly_idx, other_poly in enumerate(polyhedra):
        if other_poly_idx == poly_idx:
            continue
            
        other_verts = other_poly.get_vertices()
        
        for other_face in other_poly.get_faces():
            # Get edges of this face
            for i in range(len(other_face)):
                p1 = other_verts[other_face[i]]
                p2 = other_verts[other_face[(i + 1) % len(other_face)]]
                
                # Find intersection with our plane
                intersection = edge_plane_intersection(p1, p2, normal, d)
                
                if intersection is not None:
                    # Project to 2D
                    pt_2d = project_to_2d([intersection], origin, u, v)[0]
                    
                    # Check if inside face
                    from shapely.geometry import Point
                    if face_polygon.contains(Point(pt_2d)) or face_polygon.touches(Point(pt_2d)):
                        cutting_lines.append(pt_2d)
    
    return cutting_lines, face_polygon, origin, u, v

def subdivide_face(face_data, all_faces, polyhedra):
    """
    Subdivide a face by cutting lines from other polyhedra.
    Returns list of sub-regions (as 3D vertex lists).
    """
    face_verts = face_data['vertices_3d']
    normal = face_data['normal']
    d = face_data['d']
    poly_idx = face_data['poly_idx']
    
    # Get 2D basis
    origin = np.array(face_verts[0])
    u, v = get_plane_basis(normal)
    
    # Project face to 2D
    face_2d = project_to_2d(face_verts, origin, u, v)
    try:
        face_polygon = Polygon(face_2d)
        if not face_polygon.is_valid:
            face_polygon = make_valid(face_polygon)
        if face_polygon.is_empty:
            return []
    except:
        return []
    
    # Collect cutting lines (as 2D segments)
    cutting_segments = []
    
    for other_poly_idx, other_poly in enumerate(polyhedra):
        if other_poly_idx == poly_idx:
            continue
            
        other_verts = other_poly.get_vertices()
        
        # Check each face of other polyhedron
        for other_face in other_poly.get_faces():
            intersect_points = []
            
            # Get edges of this face
            for i in range(len(other_face)):
                p1 = other_verts[other_face[i]]
                p2 = other_verts[other_face[(i + 1) % len(other_face)]]
                
                intersection = edge_plane_intersection(p1, p2, normal, d)
                
                if intersection is not None:
                    pt_2d = project_to_2d([intersection], origin, u, v)[0]
                    intersect_points.append(pt_2d)
            
            # If we have 2 intersection points, that's a cutting line
            if len(intersect_points) >= 2:
                # Remove duplicates
                unique_pts = []
                for p in intersect_points:
                    is_dup = False
                    for q in unique_pts:
                        if abs(p[0] - q[0]) < TOL and abs(p[1] - q[1]) < TOL:
                            is_dup = True
                            break
                    if not is_dup:
                        unique_pts.append(p)
                
                if len(unique_pts) >= 2:
                    # Extend the line segment slightly to ensure it cuts through
                    p1, p2 = np.array(unique_pts[0]), np.array(unique_pts[1])
                    direction = p2 - p1
                    p1_ext = p1 - direction * 0.01
                    p2_ext = p2 + direction * 0.01
                    cutting_segments.append((tuple(p1_ext), tuple(p2_ext)))
    
    # If no cuts, return original face
    if not cutting_segments:
        return [face_verts]
    
    # Split polygon by cutting lines
    result_polygon = face_polygon
    
    for seg in cutting_segments:
        try:
            line = LineString([seg[0], seg[1]])
            if result_polygon.is_empty:
                break
            
            # Handle both single polygon and multipolygon
            if isinstance(result_polygon, (MultiPolygon, GeometryCollection)):
                new_parts = []
                for geom in result_polygon.geoms:
                    if hasattr(geom, 'exterior'):  # Is a polygon
                        try:
                            split_result = split(geom, line)
                            new_parts.extend(split_result.geoms)
                        except:
                            new_parts.append(geom)
                result_polygon = GeometryCollection(new_parts)
            else:
                try:
                    result_polygon = split(result_polygon, line)
                except:
                    pass
        except:
            continue
    
    # Extract resulting polygons
    sub_regions = []
    
    if isinstance(result_polygon, (MultiPolygon, GeometryCollection)):
        geoms = list(result_polygon.geoms)
    else:
        geoms = [result_polygon]
    
    for geom in geoms:
        if hasattr(geom, 'exterior') and not geom.is_empty:
            # Get 2D coordinates and convert back to 3D
            coords_2d = list(geom.exterior.coords)[:-1]  # Remove closing point
            coords_3d = unproject_to_3d(coords_2d, origin, u, v)
            sub_regions.append(coords_3d)
    
    return sub_regions if sub_regions else [face_verts]

def is_region_visible(region_verts, normal, poly_idx, polyhedra):
    """
    Check if a face region is visible (not inside another polyhedron).
    """
    # Get centroid
    centroid = np.mean(region_verts, axis=0)
    
    # Push slightly outward along normal
    test_point = centroid + normal * TOL * 100
    
    # Check if inside any other polyhedron
    for other_idx, other_poly in enumerate(polyhedra):
        if other_idx == poly_idx:
            continue
        
        other_verts = other_poly.get_vertices()
        other_faces = other_poly.get_faces()
        
        if point_inside_polyhedron(test_point, other_verts, other_faces):
            return False
    
    return True

def count_visible_faces(polyhedra, verbose=False):
    """
    Count visible face regions in a compound of polyhedra.
    
    Handles:
    - Face subdivision by cutting edges
    - Visibility testing (faces inside other solids are hidden)
    - Coplanar faces (merged using 2D union)
    """
    all_faces = collect_all_faces(polyhedra)
    
    # Group faces by plane for coplanar handling
    faces_by_plane = defaultdict(list)
    for face_data in all_faces:
        faces_by_plane[face_data['plane_key']].append(face_data)
    
    visible_count = 0
    hidden_count = 0
    
    # Process each plane
    for plane_key, plane_faces in faces_by_plane.items():
        if verbose:
            print(f"\nPlane {plane_key}: {len(plane_faces)} faces")
        
        # Get common 2D projection basis from first face
        first_face = plane_faces[0]
        normal = first_face['normal']
        origin = np.array(first_face['vertices_3d'][0])
        u, v = get_plane_basis(normal)
        
        if len(plane_faces) == 1:
            # Single face on this plane - subdivide and test
            face_data = plane_faces[0]
            sub_regions = subdivide_face(face_data, all_faces, polyhedra)
            
            for region in sub_regions:
                if is_region_visible(region, face_data['normal'], 
                                    face_data['poly_idx'], polyhedra):
                    visible_count += 1
                    if verbose:
                        print(f"  Region visible")
                else:
                    hidden_count += 1
                    if verbose:
                        print(f"  Region hidden")
        else:
            # Multiple coplanar faces - compute union then subdivide
            # First, union all face polygons in 2D
            polygons_2d = []
            face_poly_indices = []  # Track which polyhedron each face belongs to
            face_normals = []  # Track normals (they might point opposite ways)
            
            for face_data in plane_faces:
                face_verts = face_data['vertices_3d']
                face_2d = project_to_2d(face_verts, origin, u, v)
                try:
                    poly = Polygon(face_2d)
                    if poly.is_valid and not poly.is_empty:
                        polygons_2d.append(poly)
                        face_poly_indices.append(face_data['poly_idx'])
                        face_normals.append(face_data['normal'])
                except:
                    pass
            
            if not polygons_2d:
                continue
            
            # Compute the union of all coplanar faces
            try:
                combined = unary_union(polygons_2d)
            except:
                combined = polygons_2d[0]
            
            # Extract individual polygons from union result
            if isinstance(combined, (MultiPolygon, GeometryCollection)):
                union_polys = [g for g in combined.geoms if hasattr(g, 'exterior') and not g.is_empty]
            elif hasattr(combined, 'exterior') and not combined.is_empty:
                union_polys = [combined]
            else:
                union_polys = []
            
            # For each region in the union, check visibility
            for union_poly in union_polys:
                # Get centroid in 2D and convert to 3D
                centroid_2d = union_poly.centroid
                centroid_3d = origin + centroid_2d.x * u + centroid_2d.y * v
                
                # For coplanar faces, we need to test visibility from BOTH sides
                # A region is visible if it's exposed on at least one side
                
                # Test positive normal direction
                test_point_pos = centroid_3d + normal * TOL * 100
                visible_pos = True
                for other_idx, other_poly in enumerate(polyhedra):
                    other_verts = other_poly.get_vertices()
                    other_faces = other_poly.get_faces()
                    if point_inside_polyhedron(test_point_pos, other_verts, other_faces):
                        visible_pos = False
                        break
                
                # Test negative normal direction
                test_point_neg = centroid_3d - normal * TOL * 100
                visible_neg = True
                for other_idx, other_poly in enumerate(polyhedra):
                    other_verts = other_poly.get_vertices()
                    other_faces = other_poly.get_faces()
                    if point_inside_polyhedron(test_point_neg, other_verts, other_faces):
                        visible_neg = False
                        break
                
                # Count as visible if exposed on either side
                if visible_pos or visible_neg:
                    visible_count += 1
                    if verbose:
                        print(f"  Coplanar region visible (pos={visible_pos}, neg={visible_neg})")
                else:
                    hidden_count += 1
                    if verbose:
                        print(f"  Coplanar region hidden")
    
    if verbose:
        print(f"\nTotal: {visible_count} visible, {hidden_count} hidden")
    
    return visible_count

def analyze_compound(polyhedra, verbose=True):
    """Detailed analysis with breakdown by polyhedron"""
    visible = count_visible_faces(polyhedra, verbose=verbose)
    return {'visible_faces': visible}

def quick_count(polyhedra):
    """Quick count without output"""
    return count_visible_faces(polyhedra, verbose=False)

def count_union_faces(polyhedra):
    """
    Count faces on the boolean union of polyhedra.
    Uses trimesh for accurate CSG operations.
    
    This counts the triangular faces on the OUTER SHELL of the compound.
    """
    import trimesh
    
    meshes = []
    for poly in polyhedra:
        verts = poly.get_vertices()
        faces_list = poly.get_faces()
        
        # Triangulate faces if needed
        tri_faces = []
        for face in faces_list:
            if len(face) == 3:
                tri_faces.append(face)
            else:
                # Fan triangulation
                for i in range(1, len(face) - 1):
                    tri_faces.append([face[0], face[i], face[i+1]])
        
        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(tri_faces))
        meshes.append(mesh)
    
    if len(meshes) == 1:
        return len(meshes[0].faces)
    
    union_mesh = trimesh.boolean.union(meshes)
    return len(union_mesh.faces)

# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Polyhedra Library Test")
    print("=" * 60)
    
    # Test 1: Single octahedron
    print("\n--- Single Octahedron ---")
    oct1 = Octahedron()
    count = quick_count([oct1])
    print(f"Visible faces: {count} (expected: 8)")
    
    # Test 2: Two identical octahedra
    print("\n--- Two Identical Octahedra ---")
    oct1 = Octahedron()
    oct2 = Octahedron()
    count = quick_count([oct1, oct2])
    print(f"Visible faces: {count} (expected: 8 - coplanar)")
    
    # Test 3: Two octahedra, 45° rotation
    print("\n--- Two Octahedra, 45° rotation ---")
    oct1 = Octahedron()
    oct2 = Octahedron().rotate([0, 1, 0], 45)
    count = quick_count([oct1, oct2])
    print(f"Visible faces: {count}")
    
    # Test 4: Two octahedra, 90° rotation
    print("\n--- Two Octahedra, 90° rotation ---")
    oct1 = Octahedron()
    oct2 = Octahedron().rotate([0, 1, 0], 90)
    count = quick_count([oct1, oct2])
    print(f"Visible faces: {count} (expected: 8 - same as 0°)")
