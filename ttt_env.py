import numpy as np

class TicTacToe:
    def __init__(self):
        self.field = np.zeros((3,3))
        self.active_player = 1

    def get_actions(self):
        arrs = (self.field == 0).nonzero()
        actions = [(a, b) for a, b in zip(arrs[0], arrs[1])]
        actions = [] if self.get_done()[0] else actions
        return actions, self.active_player
    
    def get_field(self):
        return self.field
    
    def get_active_player(self):
        return self.active_player

    def get_done(self):
        done = False
        tie = False

        quer = [[], []]
        for i, j in zip(range(3), range(2, -1, -1)):
            for f in (self.field, np.swapaxes(self.field, 0, 1)):
                done = True if (f[i][0] == f[i][1] and f[i][1] == f[i][2]) and(f[i][0] == 1 or f[i][0] == 2) else done
            quer[0].append(self.field[i][i])
            quer[1].append(self.field[j][j])
        
        for i in range(2):
            done = True if (not np.any(np.array(quer[i]!=quer[i][0]))) and (quer[i][0] == 1 or quer[i][0] == 2) else done
        
        if not (self.field == 0).any():
            done = True  
            tie = True

        return done, tie

    def update_board(self, pos, player=None):
        """
        updates field 

        returns: field
        """

        if player is None:
            player = self.active_player
        self.field[pos[0]][pos[1]] = player
        return self.field

    def step(self, pos):
        self.update_board(pos)
        self.active_player = 1 if self.active_player == 2 else 1

        return self.get_done()


