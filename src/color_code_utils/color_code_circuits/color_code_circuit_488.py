from ..abstract_color_code_circuit import AbstractColorCodeCircuit
from ..color_code_tile import ColorCodeTile, draw_tiles
import stim
import math

class ColorCodeCircuit488(AbstractColorCodeCircuit):
    
    def __init__(self, distance, rounds, noise=None):
        super().__init__(distance, rounds)
        self.qubits = set()
        self.ancilla = set()
        self._tiles = self._generate_layout(distance)
        self.qtoid = dict()
        self.circuit = self._build_circuit(rounds, noise)
        
        

    def get_circuit(self):
        return self.circuit
    
    def draw_layout(self):
        if self._tiles is None:
            return
        draw_tiles(self._tiles)

    def _generate_layout(self, distance:int):
        # Implementation for 4.8.8 layout generation
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
        for _ in range(side):
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


    def _build_circuit(self, rounds, noise=None):
        """Implementation for building the 4.8.8 syndrome extraction circuit """
        circ = stim.Circuit()
        add_noise = noise is not None
        self._tiles.sort(key=lambda x:x.ancilla)
        
        for tile in self._tiles:
            self.qubits.update(q for q in tile.qubits)
            self.ancilla.add(tile.ancilla)
        sorted_q_a = sorted(self.qubits | self.ancilla)
        qa_index_map = {q:i for i,q in enumerate(sorted_q_a)}
        self.qtoid = qa_index_map
        qubit_idxs = []
        #append qubit coordinates
        for q,i in qa_index_map.items():
            circ.append("QUBIT_COORDS", [i], q)
            qubit_idxs.append(i)

        #set qubits to 0
        circ.append("R", qubit_idxs)
        if add_noise:
            circ.append("X_ERROR", qubit_idxs, noise)
        #print(f'all qubits: {self.qubits}')
        #print(f'all ancilla: {self.ancilla}')
        circ.append("TICK")

        #0th/initialisation round
        circ += self._measure_stab(qa_index_map, False, noise)

        #steady state rounds
        circ += (rounds - 1) * self._measure_stab(qa_index_map, True, noise, noise)
        #circ += self._measure_stab(qa_index_map, True)

        #final observable readout
        circ += self._measure_obs(qa_index_map, noise)

        return circ

    def _stab_measure(self, qtoimap, basis, noise=None, measure_noise=None):
        stab_circ = stim.Circuit()
        add_noise = noise is not None
        add_m_noise = measure_noise is not None
        ancilla_idxs = [qtoimap[a.ancilla] for a in self._tiles]
        all_idxs = {qtoimap[q] for q in self.qubits}
        all_idxs.update(ancilla_idxs)
        
        oct_dirs = [(-1,3),(1,3),(-3,1),(3,1),(-3,-1),(3,-1),(-1,-3),(1,-3)]
        oct0_dirs = [(-1,2),(1,2),(-3,0),(3,0),None,None,None,None]
        oct1_dirs = [None,None,None,(2,2),None,(2,0),(-2,-2),(0,-2)]
        oct2_dirs = [None,None,(-2,2),None,(-2,0),None,(0,-2),(2,-2)]
        square_dirs = [(-1,1),(1,1),(-1,-1),(1,-1)]
        if basis == 'X':
            stab_circ.append("RX", ancilla_idxs)
            if add_noise:
                stab_circ.append("Z_ERROR", ancilla_idxs, noise)
        else:
            stab_circ.append("R", ancilla_idxs)
            if add_noise:
                stab_circ.append("X_ERROR", ancilla_idxs, noise)
        stab_circ.append("TICK")
        for step in range(8):
            measurements = []
            depolarise_ops = set()
            for tile in self._tiles:
                ax,ay = tile.ancilla
                aidx = qtoimap[tile.ancilla]
                if tile.shape != 4:
                    #octagon cases
                    #check for edge case of it being a halved octagon
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
                qx, qy = ax+dx, ay+dy
                
                if (qx,qy) not in qtoimap.keys():
                    continue
                didx = qtoimap[(qx,qy)]
                if basis == 'Z':  
                    measurements.extend([didx, aidx])
                    depolarise_ops.update([didx, aidx])
                else:
                    measurements.extend([aidx, didx])
                    depolarise_ops.update([aidx, didx])
            if measurements:
                stab_circ.append("CNOT", measurements)
                if add_noise:
                    #cnot noise + idle noise
                    stab_circ.append("DEPOLARIZE2", depolarise_ops, noise)#temp replace w 2 dep 1s for 
                    #stab_circ.append("DEPOLARIZE1", depolarise_ops, noise)#horrible wont recommend
                    stab_circ.append("DEPOLARIZE1", [i for i in all_idxs if i not in depolarise_ops], noise)
                stab_circ.append("TICK")
        #measurement
        if add_m_noise:
            if basis == "Z":   
                stab_circ.append("M", ancilla_idxs, measure_noise)
                
            else:
                stab_circ.append("MX", ancilla_idxs, measure_noise)
        else:
            if basis == "Z":   
                stab_circ.append("M", ancilla_idxs)
                
            else:
                stab_circ.append("MX", ancilla_idxs)
        stab_circ.append("TICK")
        return stab_circ

    def _measure_obs(self, qtoimap, noise):
        obs_circ = stim.Circuit()
        
        # measure qubits
        sorted_qs = sorted(list(self.qubits))
        q_idxs = [qtoimap[q] for q in sorted_qs]
        obs_circ.append("M", q_idxs, noise)

        # final z-detectors
        q_rec_offsets = {q_idx : -len(q_idxs) + k for k,q_idx in enumerate(q_idxs)}
        a_index = {a:i for i,a in enumerate(self.ancilla)}
        for i, tile in enumerate(self._tiles):
            targets = [stim.target_rec(q_rec_offsets[qtoimap[q]]) for q in tile.qubits]
            a_offset = -len(q_idxs) - (len(self._tiles) - i)
            targets.append(stim.target_rec(a_offset))
            chrom_color = {'red':0,'green':1,'blue':2}[tile.color]
            chrom_annotation = chrom_color + 3
            obs_circ.append("DETECTOR", targets, [tile.ancilla[0], tile.ancilla[1], 0, chrom_annotation])

        # logical observable
        logical_z_qubits = [q for q in sorted_qs if q[1]==0]
        obs_targets = [stim.target_rec(q_rec_offsets[qtoimap[q]]) for q in logical_z_qubits]
        obs_circ.append("OBSERVABLE_INCLUDE", obs_targets, 0)
        
        return obs_circ


    def _measure_stab(self, qtoimap, include_detectors, data_noise=None, measure_noise=None):
        stab_circ = stim.Circuit()
        add_noise = data_noise is not None
        add_m_noise = measure_noise is not None
        oct_dirs = [(-1,3),(1,3),(-3,1),(3,1),(-3,-1),(3,-1),(1,-3),(-1,-3)]
        square_dirs = [(-1,1),(1,1),(-1,-1),(1,-1)]
        

        for basis in ['X', 'Z']:
            #reset ancillas
            stab_circ += self._stab_measure(qtoimap, basis, data_noise, measure_noise)
        
        if include_detectors:
            num_stabilizers = len(self._tiles) * 2
            for xz in range(2):
                for k, tile in enumerate(self._tiles):
                    ax, ay = tile.ancilla
                    chrom_color = {'red':0,'green':1,'blue':2}[tile.color]
                    chrom_annotation = chrom_color + 3*xz
                    offset = xz*len(self._tiles) + k
                    stab_circ.append("DETECTOR", [stim.target_rec(-(num_stabilizers - offset)),
                                          stim.target_rec(-num_stabilizers*2+offset)],
                                          [ax, ay, 0, chrom_annotation])
        if add_noise:
            stab_circ.append("DEPOLARIZE1", [qtoimap[q] for q in self.qubits], data_noise)

        if include_detectors:
            stab_circ.append("SHIFT_COORDS", [], [0,0,1])
        stab_circ.append("TICK")

        return stab_circ