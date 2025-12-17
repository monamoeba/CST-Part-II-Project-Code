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


        


def _TriangularPartitionIons(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int, tiles: list, qubitIonsMap: dict
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    
    # think of new heuristic 
    # try to cluster ions of same color together - cluster by shape first, then by color
    # get extra context of which ions belong to which tiles 
    #ideally want to keep track of each triangle vertex regardless of which triangle it is in to make the maths easier
    # get vertices of outer triangle to use TriangleNode clustering
    A = np.array([np.min(coords[:,0]) - 1.0, np.min(coords[:,1]) - 1.0])
    B = np.array([np.max(coords[:,0]) + 1.0, np.min(coords[:,1]) - 1.0])
    C = np.array([np.mean(coords[:,0]), np.max(coords[:,1]) + 1.0])
    all_clusters = []
    stack = deque([(np.array([A,B,C]), coords, 0)])

    while stack:
        vertices, points, depth = stack.popleft()
        
        if len(points) <= trapCapacity:
            if len(points) > 0:
                all_clusters.append(points)
                continue
        
        A, B, C = vertices

        mAB = (A + B)/2
        mBC = (B + C)/2
        mCA = (C + A)/2

        subtriangles = {'top': [A, mAB, mCA],
                        'left': [mAB, B, mBC],
                        'right': [mCA, mBC, C],
                        'center': [mAB, mBC, mCA]}

        buckets = { 'top': [], 'left': [], 'right': [], 'center': [] }

        ba = B - A
        ca = C - A
        dot00 = np.dot(ba,ba)
        dot01 = np.dot(ba, ca)
        dot11 = np.dot(ca,ca)
        denom = dot00 * dot11 - dot01 * dot01

        #constant for div
        idenom = 1.0 / denom if abs(denom) > 1e-10 else 0.0

        for p in points:
            pa = p - A
            dot20 = np.dot(pa,ba)
            dot21 = np.dot(pa,ca)

            v = (dot11 * dot20 - dot01 * dot21) / denom
            w = (dot00 * dot21 - dot01 * dot20) / denom
            u = 1.0 - v - w

            if u > 0.5:
                buckets['top'].append(p)
            elif v > 0.5:
                buckets['left'].append(p)
            elif w > 0.5:
                buckets['right'].append(p)
            else:
                buckets['center'].append(p)
            
        # add to stack
        for i in ['top', 'left', 'right', 'center']:
            if buckets[i]:
                stack.append((subtriangles[i], np.array(buckets[i]), depth + 1))
        
    return all_clusters
    # change to reduce overclustering - add some check to see if clusters can be merged together if under capacity

def _ShapePartitionions(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    pass