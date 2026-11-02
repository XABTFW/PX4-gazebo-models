#!/usr/bin/env python3
"""
Convert STL files to DAE (COLLADA) format
Requires: trimesh, pyglet, pycollada
Install: pip install trimesh pyglet pycollada numpy
"""

import trimesh
import numpy as np
import os

def convert_stl_to_dae(stl_file, dae_file, center_origin=True):
    """
    Convert STL to DAE format

    Args:
        stl_file: Input STL file path
        dae_file: Output DAE file path
        center_origin: If True, center the mesh at origin
    """
    print(f"Loading {stl_file}...")
    mesh = trimesh.load(stl_file)

    if center_origin:
        # Center the mesh at origin (important for rotation!)
        print("Centering mesh at origin...")
        mesh.vertices -= mesh.centroid

    print(f"Exporting to {dae_file}...")
    mesh.export(dae_file)
    print(f"✓ Converted: {stl_file} -> {dae_file}")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Convert all jiang STL files
    stl_files = [
        "jiang1.STL",
        "jiang2.STL",
        "jiang3.STL",
        "jiang4.STL"
    ]

    for stl_file in stl_files:
        stl_path = os.path.join(script_dir, stl_file)
        dae_file = stl_file.replace('.STL', '.dae')
        dae_path = os.path.join(script_dir, dae_file)

        if os.path.exists(stl_path):
            try:
                convert_stl_to_dae(stl_path, dae_path, center_origin=True)
            except Exception as e:
                print(f"✗ Error converting {stl_file}: {e}")
        else:
            print(f"✗ File not found: {stl_path}")

    print("\nDone! All files converted.")
