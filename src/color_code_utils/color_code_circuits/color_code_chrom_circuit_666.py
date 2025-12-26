from ..abstract_color_code_circuit import AbstractColorCodeCircuit
from ..color_code_tile import ColorCodeTile
from typing import Literal
import stim

class ColorCodeChromCircuit666(AbstractColorCodeCircuit):
    
    def __init__(self, distance, rounds, noise=None):
        super().__init__(distance, rounds)
        self.qubits = set()
        self.ancilla = set()
        self._tiles = None
        self.circuit = self._build_circuit(distance, rounds, noise)

    
    def get_circuit(self):
        return self.circuit
    def _make_color_code_tiles(self, distance:int):
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
    
    def _build_circuit(self, distance, rounds, noise=None):
        obs_basis = 'Z'
        """Creates a color code circuit with a phenomenological noise model.

        The circuit's detectors are annotated so that Chromobius can decode them.

        Args:
            obs_basis: The basis of the observable to prepare and verify at the end of the circuit.
            distance: The number of data qubits along one side of the patch.
            rounds: The number of times to apply depolarizing noise. One more than the number of
                times to apply measurement noise.
            noise: The strength of the depolarizing noise applied to the data qubits,
                and also the probability of noisy measurements reporting the wrong result.

        Returns:
            The created circuit.
        """
         
        def mpp_targets(
            qubits: list[complex],
            basis: Literal['X', 'Y', 'Z']
        ) -> list[stim.GateTarget]:
            """Makes a pauli product for an MPP instruction."""
            target_b = {'X': stim.target_x, 'Y': stim.target_y, 'Z': stim.target_z}[basis]
            indices = sorted(q2i[q] for q in qubits)
            targets = []
            for k in indices:
                targets.append(target_b(k))
                targets.append(stim.target_combiner())
            targets.pop()
            return targets

        def measure_observables() -> stim.Circuit:
            """Make instructions to measure an observable of the color code."""
            c = stim.Circuit()
            c.append("MPP", mpp_targets(sorted_qubits, obs_basis))
            c.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), 0)
            return c
            
        def measure_stabilizers(
            *,
            data_noise_after: bool,
            measure_noise: bool,
            include_detectors: bool,
        ) -> stim.Circuit:
            """Make instructions to measure the stabilizers of the color code."""
            c = stim.Circuit()

            # Measure every stabilizer.
            for basis in ['X', 'Z']:
                for tile in tiles:
                    c.append("MPP", mpp_targets(tile.qubits, basis), noise if (measure_noise and noise is not None) else None)
            
            # Compare the measurements to the previous round to produce detection events.
            if include_detectors:
                num_stabilizers = len(tiles) * 2
                for xz in range(2):
                    for k, tile in enumerate(tiles):
                        center = sum([q[0] for q in tile.qubits]) / len(tile.qubits), sum([q[1] for q in tile.qubits]) / len(tile.qubits)
                        chromobius_color = {'red': 0, 'green': 1, 'blue': 2}[tile.color]
                        chromobius_annotation = chromobius_color + xz*3
                        offset = xz * len(tiles) + k
                        c.append("DETECTOR", [
                            stim.target_rec(-num_stabilizers + offset), 
                            stim.target_rec(-num_stabilizers*2 + offset),
                        ], [center[0], center[1], 0, chromobius_annotation])
            #print(f'range len all_qubits: {range(len(all_qubits))}')
            # End the round.
            if data_noise_after and noise is not None:
                c.append("DEPOLARIZE1", [q2i[q] for q in all_qubits], noise)
            c.append("SHIFT_COORDS", [], [0, 0, 1])
            c.append("TICK")

            return c

        tiles = self._make_color_code_tiles(distance=distance)
        circuit = stim.Circuit()
        #print(f'tiles: {tiles}')

        # Index the qubit coordinates and put coordinate data in the circuit.
        all_qubits = {q for tile in tiles for q in tile.qubits}
        #print(f'all_qubits: {all_qubits}')
        sorted_qubits = sorted(all_qubits, key=lambda q: (q[0], q[1]))
        q2i = {q: i for i, q in enumerate(sorted_qubits)}
        self.qubits = set(q2i.values())
        for q, i in q2i.items():
            circuit.append("QUBIT_COORDS", [i], [q[0], q[1]])

        # Use the helper methods you just defined to build the rounds and combine them into a full circuit.
        circuit += measure_observables()
        circuit += measure_stabilizers(data_noise_after=True, measure_noise=False, include_detectors=False)
        circuit += (rounds - 1) * measure_stabilizers(data_noise_after=True, measure_noise=True, include_detectors=True)
        circuit += measure_stabilizers(data_noise_after=False, measure_noise=False, include_detectors=True)
        circuit += measure_observables()
        
        return circuit