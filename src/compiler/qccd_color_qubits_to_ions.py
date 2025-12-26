import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
    Tuple,
)
from collections import deque
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler.qccd_parallelisation import *


def TriangularPartitionIons(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    
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

            v = (dot11 * dot20 - dot01 * dot21) * idenom
            w = (dot00 * dot21 - dot01 * dot20) * idenom
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
    
    result = np.empty(len(all_clusters), dtype=Tuple[Sequence[Ion], npt.NDArray[np.float64]])
    for i in range(len(all_clusters)):
        clust = all_clusters[i]
        clusterCentre = np.mean(clust, axis=0)
        result[i] = (clust, clusterCentre)
    coordsToIons = {(c[0], c[1]): i for c, i in zip(coords, ions)}

    return result
    # change to reduce overclustering - add some check to see if clusters can be merged together if under capacity

def MergeUnderfilledClusters(
    clusters: Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]], trapCapacity: int, coordsToIons: dict
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    
    toCheck = []

    for i, (coords, center) in enumerate(clusters):
        toCheck.append({'size': len(coords), 'center': center, 'active': True, 'coords': coords})
    
    while True:
        candidates = [t for t in toCheck if t['active']]
        candidates.sort(lambda x: x[0])
        merged = False

        for i in range(len(candidates)):
            c1 = candidates[i]
            bestpartner = -1
            mindist = float('inf')

            for j in range(i+1, len(candidates)):
                c2 = candidates[j]
                dist = np.linalg.norm(c1[1] - c2[1]) 
                if dist < mindist and c1[0]+c2[0] <= trapCapacity:
                    mindist = dist
                    bestpartner = j
            if bestpartner != -1:
                c2 = candidates[bestpartner]
                
                newsize = c1[0] + c2[0]
                newcenter = (c1[0]*c1[1] + c2[0]*c2[1]) / newsize

                newcluster = {'size': newsize, 'center': newcenter, 'active': True,
                              'coords': c1['coords'] + c2['coords']}
                
                c1['active'] = False
                c2['active'] = False
                toCheck.append(newcluster)

                merged = True
                break
        if not merged:
            break
    
    finalcoords = [t['coords'] for t in toCheck if t['active']]
    res = []
    for clust in finalcoords:
        ions = [coordsToIons[(c[0], c[1])] for c in clust] 
        res.append(ions)
    return res
        
def _ShapePartitionions(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    pass