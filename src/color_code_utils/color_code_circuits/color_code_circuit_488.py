from ..abstract_color_code_circuit import AbstractColorCodeCircuit
from ..color_code_tile import ColorCodeTile, draw_tiles
import stim
import math

class ColorCodeCircuit488(AbstractColorCodeCircuit):
    
    def __init__(self, distance, rounds, noise=None, basis='Z'):
        super().__init__(distance, rounds)
        self.qubits = set()
        self.ancilla = set()
        self._tiles = self._generate_layout(distance)
        self.qtoid = dict()
        self.circuit = self._build_circuit(rounds, noise)
        self.basis = basis
        self.is_mirrored = (distance - 1) % 4 == 0
        self.circuit = self._build_circuit(noise)

    def get_circuit(self):
        return self.circuit
    
    def draw_layout(self):
        if self._tiles is None:
            return
        draw_tiles(self._tiles)

    def _generate_layout(self, distance):
        """Implementation for 4.8.8 layout generation"""
        if (distance-1)%4==0:
            layout = 1
        else:
            layout = 0
        tiles = []
        
        basenum = distance//2
        side = math.ceil(basenum/2)
        rightmost = 0
        #add the base green
        curr = [5,0]
        for _ in range(basenum):
            tile = ColorCodeTile(
                [(curr[0]+dx, curr[1]+dy) for dx,dy in [(-3,0),(-1,2),(1,2),(3,0)]],
                tuple(curr),
                'green',
                80
            )
            tiles.append(tile)
            rightmost = curr[0]+3
            curr[0] += 8
        
        curr = [2,2]
        for ind in range(side):
            tile = ColorCodeTile(
                [(curr[0]+dx, curr[1]+dy) for dx,dy in [(-2,-2),(0,-2),(2,0),(2,2)]],
                tuple(curr),
                'blue',
                81
            )
            tiles.append(tile)
            #left down
            coord = [curr[0]-1, curr[1]-3]
            issquare = True
            while coord[1]>0:
                if issquare:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-1,-1),(-1,1),(1,1),(1,-1)]],
                        tuple(coord),
                        'red',
                        4
                    )
                else:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-3,1),(-1,3),(1,3),(3,1),(3,-1),(1,-3),(-1,-3),(-3,-1)]],
                        tuple(coord),
                        'blue',
                        8
                    )
                tiles.append(tile)
                issquare = not issquare
                coord[1] -= 4

            #right down
            coord = [curr[0]+3, curr[1]+1]
            issquare = True
            while coord[1]>0:
                if issquare:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-1,-1),(-1,1),(1,1),(1,-1)]],
                        tuple(coord),
                        'red',
                        4
                    )
                else:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-3,1),(-1,3),(1,3),(3,1),(3,-1),(1,-3),(-1,-3),(-3,-1)]],
                        tuple(coord),
                        'green',
                        8
                    )
                tiles.append(tile)
                issquare = not issquare
                coord[1] -=4
            
            curr[0] += 8
            curr[1] += 8
        
        curr = [rightmost-4, 6]
        for _ in range(basenum-side):
            tile = ColorCodeTile(
                [(curr[0]+dx, curr[1]+dy) for dx,dy in [(2,-2),(0,-2),(-2,0),(-2,2)]],
                tuple(curr),
                'green',
                82
            )
            tiles.append(tile)
            #right down
            coord = [curr[0]+1, curr[1]-3]
            issquare = True
            while coord[1]>0:
                if issquare:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-1,-1),(-1,1),(1,1),(1,-1)]],
                        tuple(coord),
                        'red',
                        4
                    )
                else:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-3,1),(-1,3),(1,3),(3,1),(3,-1),(1,-3),(-1,-3),(-3,-1)]],
                        tuple(coord),
                        'green',
                        8
                    )
                tiles.append(tile)
                issquare = not issquare
                coord[1] -= 4

            #left down
            coord = [curr[0]-3, curr[1]+1]
            issquare = True
            while coord[1]>0:
                if issquare:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-1,-1),(-1,1),(1,1),(1,-1)]],
                        tuple(coord),
                        'red',
                        4
                    )
                else:
                    tile = ColorCodeTile(
                        [(coord[0]+dx, coord[1]+dy) for dx,dy in [(-3,1),(-1,3),(1,3),(3,1),(3,-1),(1,-3),(-1,-3),(-3,-1)]],
                        tuple(coord),
                        'blue',
                        8
                    )
                tiles.append(tile)
                issquare = not issquare
                coord[1] -=4
            
            curr[0] -= 8
            curr[1] += 8
        
        if layout==1:
            n = (rightmost+1)//2
            for i in range(len(tiles)):
                tiles[i].qubits = [((2*n)-x,y) for x,y in tiles[i].qubits]
                tiles[i].ancilla = (2*n - tiles[i].ancilla[0], tiles[i].ancilla[1])
                if tiles[i].color == 'green':
                    tiles[i].color = 'blue'
                elif tiles[i].color == 'blue':
                    tiles[i].color = 'green'
                if tiles[i].shape == 81:
                    tiles[i].shape = 82
                elif tiles[i].shape == 82:
                    tiles[i].shape = 81

        return tiles


    def _build_circuit(self, noise=None):
        """Build 488 color code circuit"""
        
        add_noise = noise is not None
        tiles = self._generate_layout(self.distance)
        self._tiles = tiles
        circ = stim.Circuit()

        self._tiles.sort(key=lambda x:x.ancilla)
        qubit_coords = set()
        ancilla_coords = set()
        for tile in self._tiles:
            qubit_coords.update(q for q in tile.qubits)
            ancilla_coords.add(tile.ancilla)
        
        """qubit_coords= {q for tile in tiles for q in tile.qubits}
        ancilla_coords = {tile.ancilla for tile in tiles}"""
        sorted_q_a = sorted(qubit_coords| ancilla_coords)
        qa_index_map = {q:i for i,q in enumerate(sorted_q_a)}
        chrom_annot = {'red':{'X':0, 'Z':3},'green':{'X':1, 'Z':4},'blue':{'X':2, 'Z':5}}
        self._tiles.sort(key=lambda t:qa_index_map[t.ancilla])
        self.qtoid = qa_index_map
        self.qubits = {qa_index_map[q] for q in qubit_coords}
        self.ancilla = {qa_index_map[a] for a in ancilla_coords}

        #append qubit coordinates to circuit
        for q,i in qa_index_map.items():
            circ.append("QUBIT_COORDS", [i], [q[0], q[1]])

        #reset everything
        if self.basis == 'Z':
            circ.append("R", list(self.qubits))
        else:
            circ.append("RX", list(self.qubits))
        
        if add_noise:
            circ.append("X_ERROR" if self.basis == 'Z' else "Z_ERROR",self.qubits, noise)

        circ.append("R", list(self.ancilla))
        if add_noise:
            circ.append("X_ERROR", self.ancilla, noise)
        circ.append("TICK")

        #round 0 
        circ += self._measure_round_Z_0(qa_index_map, chrom_annot, noise)

        #normal rounds
        circ += (self.rounds-1)*self._measure_rounds(qa_index_map, measures_per_round=2*len(self._tiles), chrom_annot=chrom_annot, noise=noise) #noise=noise)

        circ += self._measure_rounds(qa_index_map, measures_per_round=2*len(self._tiles), chrom_annot=chrom_annot, noise=None)
        
        # measure qubits
        sorted_qs = sorted(list(qubit_coords))
        q_idxs = [qa_index_map[q] for q in sorted_qs]

        #measure_args = [q_idxs, noise] if addnoise else [q_idxs]
        # model ideal error correction at final stage
        measure_args = [q_idxs]
        if self.basis == 'Z':
            circ.append("M", *measure_args)
        else:
            circ.append("MX", *measure_args)
        
        # final detectors
        q_rec_offsets = {q_idx : -len(q_idxs) + k for k,q_idx in enumerate(q_idxs)}
        for i, tile in enumerate(tiles):
            targets = [stim.target_rec(q_rec_offsets[qa_index_map[q]]) for q in tile.qubits]
            if self.basis == 'Z':
                a_offset = -len(q_idxs) - (len(tiles) - i)
                detector_basis = chrom_annot[tile.color]['Z']
            else:
                a_offset = -len(q_idxs) - 2*len(tiles) + i
                detector_basis = chrom_annot[tile.color]['X']
            targets.append(stim.target_rec(a_offset))

            circ.append("DETECTOR", targets, [tile.ancilla[0], tile.ancilla[1], 1, detector_basis])

        
        # logical observable
        logical_qubits = self._get_topological_logical_observable('red')
        obs_targets = [stim.target_rec(q_rec_offsets[qa_index_map[q]]) for q in logical_qubits]
        circ.append("OBSERVABLE_INCLUDE", obs_targets, 0)
        
        return circ
    
    def _get_topological_logical_observable(self, exclude_color='red'):
        """
        Generates commuting logical boundary string by finding
        all data qubits that don't belong to the excluded color's tiles.
        """
        qubits_in_excluded_color = set()
        all_data_qubits = set()
        
        for tile in self._tiles:
            all_data_qubits.update(tile.qubits)
            
            if tile.color == exclude_color:
                qubits_in_excluded_color.update(tile.qubits)
                
        logical_boundary_qubits = all_data_qubits - qubits_in_excluded_color
        
        return sorted(list(logical_boundary_qubits))

    def _measure_round_Z_0(self, qa_index_map, chrom_annot, noise=None):
        loop = stim.Circuit()
        total_m = len(self._tiles)

        self._measure_stabilizers(loop, qa_index_map, 'X', noise)
        if self.basis == 'X':
            for i,tile in enumerate(self._tiles):
                loop.append("DETECTOR", [stim.target_rec(-(total_m - i))],[tile.ancilla[0], tile.ancilla[1], 1, chrom_annot[tile.color]['X']])
        
        self._measure_stabilizers(loop, qa_index_map, 'Z', noise)
        if self.basis == 'Z':
            for i,tile in enumerate(self._tiles):
                loop.append("DETECTOR", [stim.target_rec(-(total_m - i))],[tile.ancilla[0], tile.ancilla[1], 1, chrom_annot[tile.color]['Z']])
        
        loop.append("TICK")
        loop.append("SHIFT_COORDS", [], [0,0,1])
        return loop

    def _measure_rounds(self, qa_index_map:dict, measures_per_round:int, chrom_annot:dict, noise=None):
        """Measuring the repeated rounds"""
        loop = stim.Circuit()
        add_noise = noise is not None

        #X
        self._measure_stabilizers(loop, qa_index_map, 'X', noise)
        #copied from 666
        for i, tile in enumerate(self._tiles):
            current_offset = -(len(self._tiles) - i)
            prev_offset = current_offset - measures_per_round
            loop.append("DETECTOR", [stim.target_rec(current_offset), stim.target_rec(prev_offset)], [tile.ancilla[0], tile.ancilla[1], 0, chrom_annot[tile.color]['X']])
        loop.append("TICK")
        
        #Z
        self._measure_stabilizers(loop, qa_index_map, 'Z', noise)
        for i, tile in enumerate(self._tiles):
            current_offset = -(len(self._tiles) - i)
            prev_offset = current_offset - measures_per_round
            loop.append("DETECTOR", [stim.target_rec(current_offset), 
                                     stim.target_rec(prev_offset)], [tile.ancilla[0], tile.ancilla[1], 1, chrom_annot[tile.color]['Z']])


        loop.append("TICK")
        loop.append("SHIFT_COORDS", [], [0,0,1])
        
        return loop

    def _measure_stabilizers(self, stab_circ:stim.Circuit, qa_index_map:dict, basis:str, noise=None):
        """Measuring the stabilizers"""

        add_noise = noise is not None
        all_idxs = {q for q in self.qubits}
        all_idxs.update(self.ancilla)
        
        oct_dirs = [(-1,3),(1,3),(-3,1),(3,1),(-3,-1),(3,-1),(-1,-3),(1,-3)]
        oct0_dirs = [(-1,2),(1,2),(-3,0),(3,0),None,None,None,None]
        oct0_dirs1 = [(1,2),(-1,2),(3,0),(-3,0),None,None,None,None]
        oct1_dirs = [None,None,None,(2,2),None,(2,0),(-2,-2),(0,-2)]
        oct1_dirs1 = [None,None,(2,2),None,(2,0),None,(0,-2),(-2,-2)]
        oct2_dirs = [None,None,(-2,2),None,(-2,0),None,(0,-2),(2,-2)]
        oct2_dirs1 = [None,None,None,(-2,2),None,(-2,0),(2,-2),(0,-2)]
        square_dirs = [(-1,1),(1,1),(-1,-1),(1,-1)]

        #reset ancilla qubits
        if basis == 'X':
            stab_circ.append("RX", self.ancilla)
            if add_noise:
                stab_circ.append("Z_ERROR", self.ancilla, noise)
        else:
            stab_circ.append("R", self.ancilla)
            if add_noise:
                stab_circ.append("X_ERROR", self.ancilla, noise)
        stab_circ.append("TICK")
        
        for step in range(8):
            measurements = []
            for tile in self._tiles:
                ax,ay = tile.ancilla
                aidx = self.qtoid[tile.ancilla]
                if tile.shape != 4:
                    #octagon cases
                    #check for edge case of it being a halved octagon
                    if self.is_mirrored:
                        if tile.shape == 8:
                            dirs = oct_dirs[step]
                        elif tile.shape == 80:
                            dirs = oct0_dirs1[step]
                        elif tile.shape == 81:
                            dirs = oct1_dirs1[step]
                        else:
                            dirs = oct2_dirs1[step]
                    else:
                        if tile.shape == 8:
                            dirs = oct_dirs[step]
                        elif tile.shape == 80:
                            dirs = oct0_dirs[step]
                        elif tile.shape == 81:
                            dirs = oct1_dirs[step]
                        else:
                            dirs = oct2_dirs[step]
                    if dirs is None:
                        continue
                    dx,dy = dirs
                elif tile.shape == 4 and step>=4:
                    continue
                else:
                    dx,dy = square_dirs[step]
                if (tile.shape == 8 or tile.shape ==4) and self.is_mirrored:
                    dx = -dx
                qx, qy = ax+dx, ay+dy
                
                if (qx,qy) not in self.qtoid.keys():
                    continue
                didx = self.qtoid[(qx,qy)]
                if basis == 'Z':  
                    measurements.extend([didx, aidx])
                else:
                    measurements.extend([aidx, didx])
                    
            if measurements:
                stab_circ.append("CNOT", measurements)
                
                if add_noise:
                    #cnot noise + idle noise
                    idle_qubits = [q for q in self.qubits if q not in measurements]
                    idle_ancilla = [a for a in self.ancilla if a not in measurements]
                    all_idling = idle_qubits + idle_ancilla

                    stab_circ.append("DEPOLARIZE2", measurements, noise)
                    stab_circ.append("DEPOLARIZE1", all_idling, noise)
                stab_circ.append("TICK")
                
        ancilla_idxs = [self.qtoid[a.ancilla] for a in self._tiles]
        measure_args = [ancilla_idxs, noise] if add_noise else [ancilla_idxs]
        # ancilla measurement
        if basis == "X":   
            stab_circ.append("MX", *measure_args)
    
        else:
            stab_circ.append("M", *measure_args)

        if add_noise:
            stab_circ.append("DEPOLARIZE1", self.qubits, noise)
        
        stab_circ.append("TICK")