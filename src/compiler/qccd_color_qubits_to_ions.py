import numpy as np
import numpy.typing as npt
from typing import (
    List,
    Sequence,
    Tuple,
)
from collections import deque
from scipy.spatial import cKDTree, distance_matrix
from scipy.optimize import linear_sum_assignment
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


def TriangularPartitionIons_vectorised(
    ions: Sequence[Ion],
    coords: npt.NDArray[np.float64],
    trapCapacity: int,
    merge_strategy: str = "bounded",
    merge_threshold: float = 2.5,
) -> Sequence[Tuple[Sequence[Ion], Tuple[float, float]]]:
    """
    Valid merge_strategy options: 'bounded', 'unbounded_nn', 'knn'
    """
    A = np.array([np.min(coords[:, 0]) - 1.0, np.min(coords[:, 1]) - 1.0])
    B = np.array([np.max(coords[:, 0]) + 1.0, np.min(coords[:, 1]) - 1.0])
    C = np.array([np.mean(coords[:, 0]), np.max(coords[:, 1]) + 1.0])
    all_clusters = []
    stack = deque([(np.array([A, B, C]), coords, 0)])

    while stack:
        vertices, points, depth = stack.popleft()

        if len(points) <= trapCapacity:
            if len(points) > 0:
                all_clusters.append(points)
            continue

        A, B, C = vertices

        mAB = (A + B) / 2
        mBC = (B + C) / 2
        mCA = (C + A) / 2

        subtriangles = {
            "top": [A, mAB, mCA],
            "left": [mAB, B, mBC],
            "right": [mCA, mBC, C],
            "center": [mAB, mBC, mCA],
        }

        ba = B - A
        ca = C - A
        dot00 = np.dot(ba, ba)
        dot01 = np.dot(ba, ca)
        dot11 = np.dot(ca, ca)
        denom = dot00 * dot11 - dot01 * dot01
        idenom = 1.0 / denom if abs(denom) > 1e-10 else 0.0

        # Vectorised computation of barycentric coordinates for all points in this triangle.
        # points has shape (n, 2). PA, dot20, dot21, u, v, w are all computed in one batch.
        PA = points - A  # (n, 2)
        dot20 = PA @ ba  # (n,)
        dot21 = PA @ ca  # (n,)

        v = (dot11 * dot20 - dot01 * dot21) * idenom
        w = (dot00 * dot21 - dot01 * dot20) * idenom
        u = 1.0 - v - w

        top_mask = u > 0.5
        left_mask = (~top_mask) & (v > 0.5)
        right_mask = (~top_mask) & (~left_mask) & (w > 0.5)
        center_mask = ~(top_mask | left_mask | right_mask)

        top_points = points[top_mask]
        left_points = points[left_mask]
        right_points = points[right_mask]
        center_points = points[center_mask]

        for name, subpts in [
            ("top", top_points),
            ("left", left_points),
            ("right", right_points),
            ("center", center_points),
        ]:
            if len(subpts) > 0:
                stack.append((np.array(subtriangles[name]), subpts, depth + 1))

    result = []
    for clust in all_clusters:
        clusterCentre = np.mean(clust, axis=0)
        result.append((clust, clusterCentre))

    coordsToIons = {(c[0], c[1]): i for c, i in zip(coords, ions)}

    merge_methods = {
        "bounded": lambda clusters, cap, mapping: _mergeUnderfilledClusters_kdtree(
            clusters, cap, mapping, thresh=merge_threshold
        ),
        "unbounded_nn": _merge_unbounded_nn,
        "knn": _merge_knn,
    }
    merge_func = merge_methods.get(merge_strategy)
    if merge_func is None:
        raise ValueError(
            f"Unsupported merge_strategy {merge_strategy!r}. "
            f"Valid strategies are: {', '.join(sorted(merge_methods))}"
        )

    final_clusters = merge_func(result, trapCapacity, coordsToIons)
    return final_clusters


def _mergeUnderfilledClusters_kdtree(
    clusters: Sequence[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    trapCapacity: int,
    coordsToIons: dict,
    thresh: float = 2.5,
) -> Sequence[Tuple[Sequence[Ion], Tuple[float, float]]]:
    """
    KD-tree variant of _mergeUnderfilledClusters - uses a spatial index on cluster centres to avoid exhaustive pairwise scans
    """

    # Initialise cluster records
    toCheck = []
    for coords, center in clusters:
        toCheck.append(
            {
                "size": len(coords),
                "center": np.asarray(center, dtype=float),
                "active": True,
                "coords": np.asarray(coords),
            }
        )

    while True:
        # Indices of currently active clusters
        active_indices = [i for i, c in enumerate(toCheck) if c["active"]]
        if len(active_indices) < 2:
            break

        centers = np.array([toCheck[i]["center"] for i in active_indices])
        sizes = np.array([toCheck[i]["size"] for i in active_indices])

        # Build KD-tree over current centres
        tree = cKDTree(centers)

        # Process clusters in ascending size order (prefer merging small ones)
        order = np.argsort(sizes)
        merged = False

        for ord_pos in order:
            idx0 = active_indices[ord_pos]
            c1 = toCheck[idx0]
            if not c1["active"]:
                continue

            # Query neighbours within thresh of this centre
            nbr_positions = tree.query_ball_point(c1["center"], r=thresh)
            bestpartner_global = -1
            bestdist2 = float("inf")

            for pos in nbr_positions:
                if pos == ord_pos:
                    continue
                idx_other = active_indices[pos]
                c2 = toCheck[idx_other]
                if not c2["active"]:
                    continue

                if c1["size"] + c2["size"] > trapCapacity:
                    continue

                diff = c1["center"] - c2["center"]
                dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                if dist2 < bestdist2:
                    bestdist2 = dist2
                    bestpartner_global = idx_other

            if bestpartner_global != -1:
                c2 = toCheck[bestpartner_global]

                newsize = c1["size"] + c2["size"]
                newcenter = (c1["size"] * c1["center"] + c2["size"] * c2["center"]) / newsize
                newcoords = np.concatenate((c1["coords"], c2["coords"]), axis=0)

                newcluster = {
                    "size": newsize,
                    "center": newcenter,
                    "active": True,
                    "coords": newcoords,
                }

                c1["active"] = False
                c2["active"] = False
                toCheck.append(newcluster)

                merged = True
                break

        if not merged:
            break

    finalCoordCenters = [(t["coords"], t["center"]) for t in toCheck if t["active"]]

    res = []
    for clust in finalCoordCenters:
        ions = [coordsToIons[(c[0], c[1])] for c in clust[0]]
        cent = tuple(clust[1])
        res.append((ions, cent))
    return res

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

def _merge_unbounded_nn(
    clusters: Sequence[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    trapCapacity: int,
    coordsToIons: dict,
) -> Sequence[Tuple[Sequence[Ion], Tuple[float, float]]]:

    toCheck = [
        {"size": len(coords), "center": np.asarray(center, dtype=float), "active": True, "coords": np.asarray(coords)}
        for coords, center in clusters
    ]

    while True:
        active_indices = [i for i, c in enumerate(toCheck) if c["active"]]
        if len(active_indices) < 2:
            break

        centers = np.array([toCheck[i]["center"] for i in active_indices])
        sizes = np.array([toCheck[i]["size"] for i in active_indices])
        tree = cKDTree(centers)
        order = np.argsort(sizes)
        merged = False

        for ord_pos in order:
            idx0 = active_indices[ord_pos]
            c1 = toCheck[idx0]
            if not c1["active"]:
                continue

            k_query = len(active_indices)
            distances, positions = tree.query(c1["center"], k=k_query)
            
            if k_query == 1:
                positions = [positions]

            bestpartner_global = -1

            for pos in positions:
                if pos == ord_pos:
                    continue
                
                idx_other = active_indices[pos]
                c2 = toCheck[idx_other]
                
                if c2["active"] and (c1["size"] + c2["size"] <= trapCapacity):
                    bestpartner_global = idx_other
                    break

            if bestpartner_global != -1:
                c2 = toCheck[bestpartner_global]
                newsize = c1["size"] + c2["size"]
                newcenter = (c1["size"] * c1["center"] + c2["size"] * c2["center"]) / newsize
                newcoords = np.concatenate((c1["coords"], c2["coords"]), axis=0)

                toCheck.append({
                    "size": newsize,
                    "center": newcenter,
                    "active": True,
                    "coords": newcoords,
                })

                c1["active"] = False
                c2["active"] = False
                merged = True
                break

        if not merged:
            break

    return [
        ([coordsToIons[(c[0], c[1])] for c in t["coords"]], tuple(t["center"]))
        for t in toCheck if t["active"]
    ]

def _merge_knn(
    clusters: Sequence[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    trapCapacity: int,
    coordsToIons: dict,
    k_neighbors: int = 3,
) -> Sequence[Tuple[Sequence[Ion], Tuple[float, float]]]:
    #k-nearest neighbour merging variant of _mergeUnderfilledClusters -> default 3 nearest neighbours checked if mergeable
    #ignores distance threshold and just merges with closest neighbour if merge remains <= capacity
    toCheck = [
        {"size": len(coords), "center": np.asarray(center, dtype=float), "active": True, "coords": np.asarray(coords)}
        for coords, center in clusters
    ]

    while True:
        active_indices = [i for i, c in enumerate(toCheck) if c["active"]]
        if len(active_indices) < 2:
            break

        centers = np.array([toCheck[i]["center"] for i in active_indices])
        sizes = np.array([toCheck[i]["size"] for i in active_indices])
        tree = cKDTree(centers)
        order = np.argsort(sizes)
        merged = False

        for ord_pos in order:
            idx0 = active_indices[ord_pos]
            c1 = toCheck[idx0]
            if not c1["active"]:
                continue

            k_query = min(k_neighbors + 1, len(active_indices))
            distances, positions = tree.query(c1["center"], k=k_query)

            if k_query == 1:
                positions = [positions]

            bestpartner_global = -1

            for pos in positions:
                if pos == ord_pos:
                    continue
                
                idx_other = active_indices[pos]
                c2 = toCheck[idx_other]
                
                if c2["active"] and (c1["size"] + c2["size"] <= trapCapacity):
                    bestpartner_global = idx_other
                    break

            if bestpartner_global != -1:
                c2 = toCheck[bestpartner_global]
                newsize = c1["size"] + c2["size"]
                newcenter = (c1["size"] * c1["center"] + c2["size"] * c2["center"]) / newsize
                newcoords = np.concatenate((c1["coords"], c2["coords"]), axis=0)

                toCheck.append({
                    "size": newsize,
                    "center": newcenter,
                    "active": True,
                    "coords": newcoords,
                })

                c1["active"] = False
                c2["active"] = False
                merged = True
                break

        if not merged:
            break

    return [
        ([coordsToIons[(c[0], c[1])] for c in t["coords"]], tuple(t["center"]))
        for t in toCheck if t["active"]
    ]
def regularColorPartition(measurementIons: Sequence[Ion], dataIons: Sequence[Ion], trapCapacity: int):
    """Arranges ions into clusters using regular triangular partitioning
    scheme filling to at most trapCapacity-1 ions to allow for movement in routing"""
    
    measurementIonsL = list(measurementIons)
    measurementIonCoords = np.array([list(ion.pos) for ion in measurementIonsL]).reshape(-1, 2)
    dataIonsL = list(dataIons)
    dataIonCoords = np.array([list(ion.pos) for ion in dataIonsL]).reshape(-1, 2)

    ids = measurementIonsL + dataIonsL
    coords = np.concatenate([measurementIonCoords, dataIonCoords])

    clusters = TriangularPartitionIons(ids, coords, trapCapacity-1)

    return clusters


def regularColorPartition_vectorised(
    measurementIons: Sequence[Ion],
    dataIons: Sequence[Ion],
    trapCapacity: int,
    merge_strategy: str = "bounded",
    merge_threshold: float = 2.5,
):
    measurementIonsL = list(measurementIons)
    measurementIonCoords = np.array([list(ion.pos) for ion in measurementIonsL]).reshape(-1, 2)
    dataIonsL = list(dataIons)
    dataIonCoords = np.array([list(ion.pos) for ion in dataIonsL]).reshape(-1, 2)

    ids = measurementIonsL + dataIonsL
    coords = np.concatenate([measurementIonCoords, dataIonCoords])

    clusters = TriangularPartitionIons_vectorised(
        ids,
        coords,
        trapCapacity - 1,
        merge_strategy=merge_strategy,
        merge_threshold=merge_threshold,
    )

    return clusters
      
def directCoordinateMapping(
    clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
    allGridPos: Sequence[Tuple[int, int]],
    margin: float = 2.0,
) -> List[Tuple[int, int]]:
    
    A = np.array([c[1] for c in clusters], dtype=float)   # (n_clusters, 2)
    B = np.array(allGridPos, dtype=float)                  # (n_positions, 2)

    min_coord = A.min(axis=0) - margin
    max_coord = A.max(axis=0) + margin
    in_box = np.all((B >= min_coord) & (B <= max_coord), axis=1)
    candidate_indices = np.where(in_box)[0]

    if len(candidate_indices) < len(A):
        candidate_indices = np.arange(len(B))

    B_candidates = B[candidate_indices]
    cost_matrix = distance_matrix(A, B_candidates)
    _, col_ind = linear_sum_assignment(cost_matrix)
    return [allGridPos[candidate_indices[j]] for j in col_ind]