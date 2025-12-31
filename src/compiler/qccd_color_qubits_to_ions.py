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
) -> Sequence[Tuple[Sequence[Ion], Tuple[float, float]]]:
    
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
    
    #print(f'after triangular clusters: {all_clusters}')
    result = []
    for i in range(len(all_clusters)):
        clust = all_clusters[i]
        clusterCentre = np.mean(clust, axis=0)
        result.append((clust, clusterCentre))
    coordsToIons = {(c[0], c[1]): i for c, i in zip(coords, ions)}
    final_clusters = _mergeUnderfilledClusters(result, trapCapacity, coordsToIons)
    #print(f'final clusters: {final_clusters}')
    return final_clusters
    # change to reduce overclustering - add some check to see if clusters can be merged together if under capacity

def _mergeUnderfilledClusters(
    clusters: Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]], trapCapacity: int, coordsToIons: dict
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    
    toCheck = []

    for i, (coords, center) in enumerate(clusters):
        toCheck.append({'size': len(coords), 'center': center, 'active': True, 'coords': coords})
    #distance threshold value - currently considering only 6.6.6 plaquette
    #TODO change to accomodate threshold for diff tesselations
    thresh = 2.5

    while True:
        
        candidates = [t for t in toCheck if t['active']]
        candidates.sort(key=lambda x: x['size'])
        merged = False
        #print(f'candidates = {candidates}')
        for i in range(len(candidates)):
            c1 = candidates[i]
            bestpartner = -1
            mindist = float('inf')

            for j in range(i+1, len(candidates)):
                c2 = candidates[j]
                dist = np.linalg.norm(c1['center'] - c2['center'])
                # avoid merging clusters too far away e.g. qubits not in same plaquette
                if dist < mindist and dist<=thresh and c1['size']+c2['size'] <= trapCapacity:
                    mindist = dist
                    bestpartner = j
            if bestpartner != -1:
                c2 = candidates[bestpartner]
                
                newsize = c1['size'] + c2['size']
                newcenter = (c1['size']*c1['center'] + c2['size']*c2['center']) / newsize

                newcluster = {'size': newsize, 'center': newcenter, 'active': True,
                              'coords': np.concatenate((c1['coords'], c2['coords']), axis=0)}
                
                c1['active'] = False
                c2['active'] = False
                toCheck.append(newcluster)

                merged = True
                break
        if not merged:
            break
    
    finalCoordCenters = [(t['coords'],t['center']) for t in toCheck if t['active']]

    #print(f'final coords: {finalcoords}')
    res = []
    for clust in finalCoordCenters:
        ions = [coordsToIons[(c[0], c[1])] for c in clust[0]] 
        cent = tuple(clust[1])
        res.append((ions,cent))
    return res

def regularColorPartition(measurementIons: Sequence[Ion], dataIons: Sequence[Ion], trapCapacity: int):
    dIonsPerTrap = trapCapacity
    measurementIonsL = list(measurementIons)
    measurementIonCoords = np.array([list(ion.pos) for ion in measurementIonsL])
    dataIonsL = list(dataIons)
    dataIonCoords = np.array([list(ion.pos) for ion in dataIonsL])

    ids = measurementIonsL + dataIonsL
    coords = np.concatenate([measurementIonCoords, dataIonCoords])

    clusters = TriangularPartitionIons(ids, coords, dIonsPerTrap)

    return clusters

def _ShapePartitionions(
    ions: Sequence[Ion], coords: npt.NDArray[np.float64], trapCapacity: int
) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float64]]]:
    pass