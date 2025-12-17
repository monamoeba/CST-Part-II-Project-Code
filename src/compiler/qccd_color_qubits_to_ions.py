import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
    Tuple,
)
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler.qccd_parallelisation import *

class TriangleNode:
    def __init__(self, vertices, points, depth, capacity, collect=None):
        self.vertices = vertices
        self.points = points
        self.depth = depth
        self.capacity = capacity
        self.children: List[TriangleNode] = [] 
        self.clusters = [points]

        if collect is None:
            self.collect = []
        else:
            self.collect = collect
        
        if len(self.points) > self.capacity:
            self._subdivide()
        else:
            self.collect.append(self.points)
        

    def _subdivide(self):
        A, B, C = self.vertices
        mAB = (A + B)/2
        mBC = (B + C)/2
        mCA = (C + A)/2

        top = [A, mAB, mCA]
        left = [mAB, B, mBC]
        right = [mCA, mBC, C]
        center = [mAB, mBC, mCA]

        buckets = { 'top': [], 'left': [], 'right': [], 'center': [] }

        for p in self.points:
            wA, wB, wC = self._barycentric(p,A,B,C)

            if wA > 0.5:
                buckets['top'].append(p)
            elif wB > 0.5:
                buckets['left'].append(p)
            elif wC > 0.5:
                buckets['right'].append(p)
            else:
                buckets['center'].append(p)
        
        subtris = [top, left, right, center]
        for i in range(4):
            vert = subtris[i]
            pts = buckets[list(buckets.keys())[i]]
            if pts:
                childNode = TriangleNode(vert, pts, self.depth + 1, self.capacity, self.collect)
                self.children.append(childNode)

    def _barycentric(self, point, a,b,c):

        ba = b - a
        ca = c - a
        pa = point - a

        dot00 = np.dot(ba,ba)
        dot01 = np.dot(ba, ca)
        dot11 = np.dot(ca,ca)
        dot20 = np.dot(pa,ba)
        dot21 = np.dot(pa,ca)

        denom = dot00 * dot11 - dot01 * dot01

        if abs(denom) < 1e-10:
            return np.array([0,0,0]) 
        
        v = (dot11 * dot20 - dot01 * dot21) / denom
        w = (dot00 * dot21 - dot01 * dot20) / denom
        u = 1.0 - v - w

        return np.array([u,v,w])
        


def _TriangularPartitionIons(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int, tiles: list, qubitIonsMap: dict
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    
    # think of new heuristic 
    # try to cluster ions of same color together - cluster by shape first, then by color
    # get extra context of which ions belong to which tiles 
    triangles = [list(coords)]
    #ideally want to keep track of each triangle vertex regardless of which triangle it is in to make the maths easier

    # get vertices of outer triangle to use TriangleNode clustering
    pass

def _ShapePartitionions(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    pass