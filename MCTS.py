import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, 0)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = np.array([self.Nsa.get((s,a), 0) for a in range(self.game.getActionSize())])

        if temp == 0:
            probs = np.zeros(shape=(len(counts)), dtype=int)
            probs[np.random.choice(np.flatnonzero(counts == counts.max()))] = 1
            return probs

        if temp != 1:
            counts = np.power(counts, (1./temp))                    # [x**(1./temp) for x in counts]
        probs = np.divide(counts, np.sum(counts), dtype=np.single)  # [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, canonicalBoard, depth):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=0:
            # terminal node
            # print("Terminal Node, won={}", -self.Es[s])
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            if isinstance(v, np.ndarray):
                assert len(v) == 1
                v = v[0]
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = np.multiply(self.Ps[s], valids, dtype=self.Ps[s].dtype)      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        if depth >= self.args.maxMCTSMoveDepth:
            # max move depth reached
            #print("max move depth reached")
            return 0

        valids = self.Vs[s]
        e = self.args.epsilon if depth == 0 else 0
        if e > 0:
            noise = np.random.dirichlet(np.full(len(valids), self.args.dirAlpha))
        Us = np.full(len(valids), -math.inf)
        has_valids = False
        for a in np.flatnonzero(valids):
            a_noise = noise[a] if e > 0 else 0
            Us[a] = self._calc_upper_confidence(a, s, e, a_noise)
            has_valids = True
        assert has_valids
        # choose one of the actions that has a value equal to the maximum
        best_act = np.random.choice(np.flatnonzero(Us == Us.max()))
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, depth=depth+1)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

    def _calc_upper_confidence(self, a, s, e=0, noise=0):
        if (s, a) in self.Qsa:
            q = self.Qsa[(s, a)]
            n_s_a = self.Nsa[(s, a)]
            ns = self.Ns[s]
        else:
            q = 0
            n_s_a = 0
            ns = self.Ns[s]
            if not e:
                ns += EPS
        p = self.Ps[s][a]
        if noise and e > 0:
            p = (1 - e) * p + e * noise

        u = q + self.args.cpuct * p * math.sqrt(ns) / (1 + n_s_a)
        return u
