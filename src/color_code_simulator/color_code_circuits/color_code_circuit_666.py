from ..abstract_color_code_circuit import AbstractColorCodeCircuit
from ..color_code_tile import ColorCodeTile
import stim

class ColorCodeCircuit666(AbstractColorCodeCircuit):

    #TODO: fix generation logic for distances >=7 (e.g. ancilla (10,8) at dist 7 missing)
    def generate_layout(self):
        side = self.distance*3 - 2
        xtype = [['D', 'D', 'M'], ['M','D','D'], ['D','M','D']]
        dirs = [(-1,-1), (1,-1),(-2,0),(2,0),(-1,1),(1,1)]
        tiles = []
        validpos = set()
        for y in range(0, side):
            #print(f'index = {y%3}, xtype = {xtype}')
            xpattern = xtype[y % 3]
            patternptr = 0
            for x in range(y,side-y,2):
                print(x,y)
                currtype = xpattern[patternptr % 3]
                if currtype == 'M':
                    tile = ColorCodeTile(
                        qubits = [(x+dx, y+dy) for dx,dy in dirs if self.within_bounds(x+dx,y+dy,side)],
                        ancilla = (x,y),
                        color = ['red','green','blue'][y % 3])
                    tiles.append(tile)
                patternptr += 1
        #print(f"tiles = {tiles}")
        return tiles

    def within_bounds(self,x,y,side):
        if y<0:
            return False
        if x<y:
            return False
        if y>(side-1)-x:
            return False
        return True

    def build_circuit(self):
        pass
