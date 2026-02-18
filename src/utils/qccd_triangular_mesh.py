from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Dict,
)
import math

class TriMesh:
    """
    A triangular mesh class for generating the positions of traps and junctions 
    Assumes width and height are odd to simplify cluster assignment using an expanding hexagon technique
    """
    def __init__(self, width:int, height:int, padding:int = 1):
        self.w = width
        self.h = height
        self.trap_pos = []
        self.junc_pos = []
        self.mesh_edges = []
        self.trap_count = width*height - math.floor(height/2)

    def generate_positions(self):
        if self.w % 2 == 0 or self.h % 2 == 0:
            raise ValueError("Width and/or height not odd")
        
        gridPos = []
        for j in range(self.h):
            offset = j % 2
            for i in range(self.w):
                gridPos.append((2*i + offset, 2*j))
        
        