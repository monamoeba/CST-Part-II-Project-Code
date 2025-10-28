
from abc import ABC, abstractmethod
import stim

class AbstractColorCodeCircuit:
    def __init__(self, distance: int, rounds: int):
        if distance % 2 ==0 or distance <3:
            raise ValueError("Requires odd distance >=3")
        self.distance = distance
        self.rounds = rounds

    @abstractmethod
    def generate_layout(self):
        '''Calculates positioning of data + ancilla qubits using distance'''
        pass

    @abstractmethod
    def build_circuit(self):
        '''Creates the stim Circuit representation'''
        pass

    def get_circuit(self) -> stim.Circuit:
        self.generate_layout()

        circuit = self.build_circuit()

        return circuit
    
