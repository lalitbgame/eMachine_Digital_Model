# -*- coding: mbcs -*-
"""
ABAQUS ODB postprocessing script
--------------------------------
Creates radial paths at different axial locations of a rotor lamination
and extracts deformation from all steps / all frames into a CSV file.

Works directly with ODB nodal data, which is generally more robust than
session-based path extraction for automated batch processing.

Author notes:
- Assumes rotor axis is global Z
- Assumes radial paths are rays in XY plane
- Assumes displacement field output U exists
- Compatible with ABAQUS Python (Python 2.7 style)

Usage example:
abaqus python extract_rotor_radial_deformation.py

Edit the USER INPUTS section before running.
"""

from odbAccess import openOdb
from abaqusConstants import NODAL
import csv
import math
import os
import sys


# ============================================================
# USER INPUTS
# ============================================================

ODB_PATH = r"your_model.odb"
INSTANCE_NAME = "ROTOR_LAMINATION-1"     # ODB instance name exactly as in odb.rootAssembly.instances.keys()

CSV_OUTPUT = r"radial_deformation_output.csv"

# Rotor center in global coordinates
CENTER_X = 0.0
CENTER_Y = 0.0

# Axial locations (global Z) where radial paths will be created
AXIAL_Z_LOCATIONS = [0.0, 10.0, 20.0]

# Angular locations of radial paths in degrees
# Example: [0.0] means one radial path along +X direction
# Example: [0.0, 90.0, 180.0, 270.0] means four directions
THETA_DEG_LIST = [0.0]

# Tolerances
Z_TOL = 0.25          # node is considered on axial slice if abs(z - target_z) <= Z_TOL
ANG_TOL_DEG = 5.0     # node is considered on path if angular difference <= ANG_TOL_DEG

# Minimum / maximum radius filter for the path
R_MIN = 0.0
R_MAX = 1.0e20

# If True, output only nodes in increasing radius order along the path
SORT_BY_RADIUS = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def angle_deg_from_xy(x, y, cx, cy):
    """Return angle in degrees [0, 360) of point (x, y) around center (cx, cy)."""
    ang = math.degrees(math.atan2(y - cy, x - cx))
    if ang < 0.0:
        ang += 360.0
    return ang


def angular_difference_deg(a1, a2):
    """Minimum absolute angular difference between two angles in degrees."""
    diff = abs(a1 - a2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def radial_distance(x, y, cx, cy):
    """Radial distance in XY plane from center."""
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def safe_get_disp_map(frame, instance):
    """
    Build a map: nodeLabel -> (u1, u2, u3)
    using displacement field U at NODAL position for the given instance.
    """
    if 'U' not in frame.fieldOutputs.keys():
        raise RuntimeError("Displacement field output 'U' not found in frame.")

    u_field = frame.fieldOutputs['U']
    u_sub = u_field.getSubset(region=instance, position=NODAL)

    disp_map = {}
    for v in u_sub.values:
        data = v.data
        # Ensure 3 components
        if len(data) == 3:
            disp_map[v.nodeLabel] = (data[0], data[1], data[2])
        elif len(data) == 2:
            disp_map[v.nodeLabel] = (data[0], data[1], 0.0)
        else:
            # Fallback for unexpected data shapes
            padded = list(data) + [0.0, 0.0, 0.0]
            disp_map[v.nodeLabel] = (padded[0], padded[1], padded[2])

    return disp_map


def select_nodes_on_radial_path(instance, target_z, theta_deg, cx, cy,
                                z_tol, ang_tol_deg, rmin, rmax):
    """
    Select nodes belonging to a radial path defined by:
    - axial plane z = target_z
    - angular direction theta_deg
    """
    selected = []

    for node in instance.nodes:
        x, y, z = node.coordinates

        # Axial filter
        if abs(z - target_z) > z_tol:
            continue

        # Radius filter
        r = radial_distance(x, y, cx, cy)
        if r < rmin or r > rmax:
            continue

        # Angle filter
        ang = angle_deg_from_xy(x, y, cx, cy)
        dang = angular_difference_deg(ang, theta_deg)
        if dang > ang_tol_deg:
            continue

        selected.append((node.label, x, y, z, r, ang, dang))

    if SORT_BY_RADIUS:
        selected.sort(key=lambda item: item[4])

    return selected


def compute_radial_displacement(x, y, cx, cy, u1, u2):
    """
    Radial displacement = projection of in-plane displacement on local radial direction.
    """
    dx = x - cx
    dy = y - cy
    r = math.sqrt(dx * dx + dy * dy)
    if r < 1.0e-16:
        return 0.0
    erx = dx / r
    ery = dy / r
    return u1 * erx + u2 * ery


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(ODB_PATH):
        raise IOError("ODB file not found: %s" % ODB_PATH)

    print("Opening ODB: %s" % ODB_PATH)
    odb = openOdb(path=ODB_PATH, readOnly=True)

    try:
        asm = odb.rootAssembly

        if INSTANCE_NAME not in asm.instances.keys():
            raise KeyError("Instance '%s' not found in ODB. Available instances: %s" %
                           (INSTANCE_NAME, asm.instances.keys()))

        instance = asm.instances[INSTANCE_NAME]

        # ------------------------------------------------------------
        # Precompute path-node groups once from undeformed coordinates
        # ------------------------------------------------------------
        path_definitions = []   # list of dicts

        print("Selecting nodes for radial paths...")
        for zloc in AXIAL_Z_LOCATIONS:
            for theta_deg in THETA_DEG_LIST:
                nodes_on_path = select_nodes_on_radial_path(
                    instance=instance,
                    target_z=zloc,
                    theta_deg=theta_deg,
                    cx=CENTER_X,
                    cy=CENTER_Y,
                    z_tol=Z_TOL,
                    ang_tol_deg=ANG_TOL_DEG,
                    rmin=R_MIN,
                    rmax=R_MAX
                )

                if len(nodes_on_path) == 0:
                    print("WARNING: No nodes found for z=%.6f, theta=%.3f deg" % (zloc, theta_deg))
                else:
                    print("Path found: z=%.6f, theta=%.3f deg, nodes=%d" %
                          (zloc, theta_deg, len(nodes_on_path)))

                path_definitions.append({
                    'zloc': zloc,
                    'theta_deg': theta_deg,
                    'nodes': nodes_on_path
                })

        # ------------------------------------------------------------
        # Write output CSV
        # ------------------------------------------------------------
        with open(CSV_OUTPUT, 'wb') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                'step_name',
                'frame_id',
                'frame_value',
                'path_z',
                'path_theta_deg',
                'path_node_index',
                'node_label',
                'x0',
                'y0',
                'z0',
                'r0',
                'angle_deg',
                'u1',
                'u2',
                'u3',
                'umag',
                'urad'
            ])

            # --------------------------------------------------------
            # Loop through all steps and all frames
            # --------------------------------------------------------
            for step_name in odb.steps.keys():
                step = odb.steps[step_name]
                print("\nProcessing step: %s" % step_name)

                for frame_id, frame in enumerate(step.frames):
                    print("  Frame %d / %d" % (frame_id + 1, len(step.frames)))

                    # Build displacement map for this frame
                    disp_map = safe_get_disp_map(frame, instance)

                    # For each path
                    for pdef in path_definitions:
                        zloc = pdef['zloc']
                        theta_deg = pdef['theta_deg']
                        path_nodes = pdef['nodes']

                        for i, node_info in enumerate(path_nodes):
                            node_label, x0, y0, z0, r0, ang_deg, dang = node_info

                            if node_label not in disp_map:
                                # Missing node displacement in this frame; skip
                                continue

                            u1, u2, u3 = disp_map[node_label]
                            umag = math.sqrt(u1 * u1 + u2 * u2 + u3 * u3)
                            urad = compute_radial_displacement(x0, y0, CENTER_X, CENTER_Y, u1, u2)

                            writer.writerow([
                                step_name,
                                frame_id,
                                frame.frameValue,
                                zloc,
                                theta_deg,
                                i + 1,
                                node_label,
                                x0,
                                y0,
                                z0,
                                r0,
                                ang_deg,
                                u1,
                                u2,
                                u3,
                                umag,
                                urad
                            ])

        print("\nDone. CSV saved to:")
        print(CSV_OUTPUT)

    finally:
        odb.close()
        print("ODB closed.")


if __name__ == "__main__":
    main()
