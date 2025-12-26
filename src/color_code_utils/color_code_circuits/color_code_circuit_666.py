from ..abstract_color_code_circuit import AbstractColorCodeCircuit
from ..color_code_tile import ColorCodeTile, draw_tiles
import stim

class ColorCodeCircuit666(AbstractColorCodeCircuit):
    
    def __init__(self, distance, rounds, noise=None):
        super().__init__(distance, rounds)
        self.qubits = set()
        self.ancilla = set()
        self._tiles = None
        self.all_coords = set()
        self.circuit = self._build_circuit(noise)
        
        

    def get_circuit(self):
        return self.circuit
    
    def draw_layout(self):
        if self._tiles is None:
            return
        draw_tiles(self._tiles)

    def _generate_layout(self, distance:int):
        side = distance*3 - 2
        xtype = [['D', 'D', 'M'], ['M','D','D'], ['D','M','D']]
        dirs = [(-1,-1), (1,-1),(2,0),(1,1),(-1,1),(-2,0)]
        tiles = []
        for y in range(0, side):
            xpattern = xtype[y % 3]
            patternptr = 0
            for x in range(y,side-y,2):
                currtype = xpattern[patternptr % 3]
                if currtype == 'M':
                    #color_idx = (x//2 + y) % 3
                    tile_color = ['red','green','blue'][(x+y) % 3]
                    #tile_color = ['red','green','blue'][color_idx]
                    tile = ColorCodeTile(
                        qubits = [(x+dx, y+dy) for dx,dy in dirs if self._within_bounds(x+dx,y+dy,side)],
                        ancilla = (x,y),
                        color = tile_color)
                    tiles.append(tile)
                patternptr += 1
        return tiles

    def _within_bounds(self,x,y,side):
        if y<0:
            return False
        if x<y:
            return False
        if y>(side-1)-x:
            return False
        return True

    def _build_circuit(self, noise=None):
        if noise is not None:
            addnoise = True
        else:
            addnoise = False
        tiles = self._generate_layout(self.distance)
        self._tiles = tiles
        circ = stim.Circuit()

        qubitsCoords = {q for tile in tiles for q in tile.qubits}
        ancillaCoords = {tile.ancilla for tile in tiles}
        sorted_q_a = sorted(qubitsCoords | ancillaCoords)
        self.all_coords = set(sorted_q_a)
        qa_index_map = {q:i for i,q in enumerate(sorted_q_a)}
        chrom_annot = {'red':{'X':0, 'Z':3},'green':{'X':1, 'Z':4},'blue':{'X':2, 'Z':5}}
        self.qubits = {qa_index_map[q] for q in qubitsCoords}
        self.ancilla = {qa_index_map[a] for a in ancillaCoords}
        
        # append coords to the circuit
        for q,i in qa_index_map.items():
            circ.append("QUBIT_COORDS", [i], [q[0], q[1]])

        # reset everything
        circ.append("R", list(self.qubits))
        if addnoise:
            circ.append("X_ERROR",self.qubits, noise)
        circ.append("R", list(self.ancilla))
        if addnoise:
            circ.append("X_ERROR", self.ancilla, noise)
        circ.append("TICK")
        
        # round 0:
        #self._measure_stabilizers(circ, tiles, qa_index_map, 'X')
        circ += self._measure_round_Z_0(tiles, qa_index_map, chrom_annot=chrom_annot, noise=noise)

        
        circ += (self.rounds-1) * self._measure_rounds(tiles, qa_index_map, measures_per_round=2*len(tiles), chrom_annot=chrom_annot, noise=noise)

        # measure qubits
        sorted_qs = sorted(list(qubitsCoords))
        q_idxs = [qa_index_map[q] for q in sorted_qs]
        circ.append("M", q_idxs)
        
        # final z-detectors
        q_rec_offsets = {q_idx : -len(q_idxs) + k for k,q_idx in enumerate(q_idxs)}
        for i, tile in enumerate(tiles):
            targets = [stim.target_rec(q_rec_offsets[qa_index_map[q]]) for q in tile.qubits]
            a_offset = -len(q_idxs) - (len(tiles) - i)
            targets.append(stim.target_rec(a_offset))

            circ.append("DETECTOR", targets, [tile.ancilla[0], tile.ancilla[1], 1, chrom_annot[tile.color]['Z']])

        # logical observable
        logical_z_qubits = [q for q in sorted_qs if q[1]==0]
        obs_targets = [stim.target_rec(q_rec_offsets[qa_index_map[q]]) for q in logical_z_qubits]
        circ.append("OBSERVABLE_INCLUDE", obs_targets, 0)
        
        return circ

    def _measure_round_Z_0(self, tiles:dict, qa_index_map:dict, chrom_annot:dict, noise=None):
        loop = stim.Circuit()
        total_m  = len(tiles)

        self._measure_stabilizers(loop,tiles,qa_index_map,'X', noise)
        self._measure_stabilizers(loop,tiles,qa_index_map,'Z', noise)
        for i, tile in enumerate(tiles):
            loop.append("DETECTOR", [stim.target_rec(-(total_m - i))],[tile.ancilla[0], tile.ancilla[1], 1, chrom_annot[tile.color]['Z']])
        if noise is not None:
            loop.append("Z_ERROR", self.ancilla, noise)
            loop.append("X_ERROR", self.ancilla, noise)
        loop.append("TICK")
        loop.append("SHIFT_COORDS", [], [0,0,1])
        return loop
    
    def _measure_rounds(self, tiles:dict, qa_index_map:dict, measures_per_round:int, chrom_annot:dict, noise=None):
        loop = stim.Circuit()
        if noise is not None:
            addnoise = True
        else:
            addnoise = False
        
        # X 
        self._measure_stabilizers(loop,tiles,qa_index_map, 'X', noise)
        for i, tile in enumerate(tiles):
            current_offset = -(len(tiles) - i)
            prev_offset = current_offset - measures_per_round
            loop.append("DETECTOR", [stim.target_rec(current_offset), stim.target_rec(prev_offset)], [tile.ancilla[0], tile.ancilla[1], 0, chrom_annot[tile.color]['X']])
        if addnoise:
            loop.append("Z_ERROR", self.ancilla, noise)
        loop.append("TICK")
        
        # Z
        self._measure_stabilizers(loop,tiles,qa_index_map, 'Z', noise)
        for i, tile in enumerate(tiles):
            current_offset = -(len(tiles) - i)
            prev_offset = current_offset - measures_per_round
            loop.append("DETECTOR", [stim.target_rec(current_offset), 
                                     stim.target_rec(prev_offset)], [tile.ancilla[0], tile.ancilla[1], 1, chrom_annot[tile.color]['Z']])
        if addnoise:
            loop.append("X_ERROR", self.ancilla, noise)
        loop.append("TICK")
        loop.append("SHIFT_COORDS", [], [0,0,1])
        return loop

    def _measure_stabilizers(self, circuit:stim.Circuit, tiles:dict, qa_index_map:dict, basis:str, noise=None):
        if noise is not None:
            addnoise = True
        else:
            addnoise = False
        # group by edge directions
        # order based on https://www.researchgate.net/publication/384079892_Improving_Threshold_for_Fault-Tolerant_Color-Code_Quantum_Computing_by_Flagged_Weight_Optimization
        measure_dirs = [(-1,-1),(1,-1),(-2,0),(2,0),(-1,1),(1,1)]
        #measure_dirs = [(1,-1),(2,0),(-1,-1),(1,1),(-2,0),(-1,1)]

        #TODO: add in flag gadgets implementation for reducing error-propagation paths
        # collect all edges doing CNOTs in these dirs + do them together
        measured_a = []
        cnot_groups = {i:[] for i in range(6)}
        for tile in tiles:
            (a1,a2) = tile.ancilla
            a_idx = qa_index_map[tile.ancilla]
            measured_a.append(a_idx)
            for i in range(6):
                (dx,dy) = measure_dirs[i]
                q = (a1+dx,a2+dy)
                if q in tile.qubits:
                    q_idx = qa_index_map[q]
                    if basis == 'X':
                        cnot_groups[i].extend([a_idx, q_idx])
                    else:
                        cnot_groups[i].extend([q_idx, a_idx])
        if basis == 'X':
            circuit.append("RX", measured_a)
            if addnoise:
                circuit.append("Z_ERROR", measured_a, noise)
        else:
            circuit.append("R", measured_a)
            if addnoise:
                circuit.append("X_ERROR", measured_a, noise)
        for i,group in cnot_groups.items():
            circuit.append("CNOT", group)
            if addnoise:
                circuit.append("DEPOLARIZE2", group, noise)
                #get unCNOTTED qubits in this operation
                circuit.append("DEPOLARIZE1", [q for q in self.qubits if q not in group], noise)
            circuit.append("TICK")
        if basis=='X':
            circuit.append("MX", measured_a)
        else:
            circuit.append("M", measured_a)
        if addnoise:
            circuit.append("DEPOLARIZE1", self.qubits, noise)


    def add_noise_model_to_circuit(self):
        pass