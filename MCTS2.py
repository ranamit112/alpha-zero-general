#based on jl1990/alpha-zero-general/master/MCTS.py, revision 1ee385f, March 22 2019

import hashlib
import sys
import time
import math
import numpy as np

EPS = 1e-8
#sys.setrecursionlimit(100000)


class Node:
    def __init__(self, mcts, board, ident):
        self.board = board
        self.edges = []
        self.mcts = mcts
        self.id = ident
        self.validsP1 = None
        self.validsP2 = None

    def getValids(self, player):
        if player == 1:
            if self.validsP1 is None:
                self.validsP1 = self.mcts.game.getValidMoves(self.board, player)
            return self.validsP1
        else:
            if self.validsP2 is None:
                self.validsP2 = self.mcts.game.getValidMoves(self.board, player)
            return self.validsP2

    def isLeaf(self, player):
        return not any([player == edge.player for edge in self.edges])

    def isRootNode(self):
        return self.mcts.root.id == self.id

    def addEdge(self, outNode, player, prior, action):
        self.edges.append(Edge(self, outNode, player, prior, action))

    def solveLeaf(self, player):
        if self not in self.mcts.Es:
            self.mcts.Es[self.id] = self.mcts.game.getGameEnded(self.board, player)
        if self.mcts.Es[self.id] != 0:
            return self.mcts.Es[self.id]  # terminal node
        probs, v = self.mcts.nnet.predict(self.board)
        valids = self.getValids(player)
        probs = np.multiply(probs, valids, dtype=probs.dtype)  # masking invalid moves
        sum_Ps_s = np.sum(probs)
        if sum_Ps_s > 0:
            probs /= sum_Ps_s
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            print("All valid moves were masked, do workaround.")
            probs = probs + valids
            probs /= np.sum(probs)

        for a in np.flatnonzero(valids):
            next_s, next_player = self.mcts.game.getNextState(self.board, player, a)
            stateId = self.mcts.calculateId(self.mcts.game, next_s)
            if stateId in self.mcts.tree:
                node = self.mcts.tree[stateId]
            else:
                node = Node(self.mcts, next_s, stateId)
                self.mcts.addNode(node, stateId)
            self.addEdge(node, player, probs[a], a)
        return v

    def expand(self, player):
        currentNode = self
        backfillEdges = []
        currentPlayer = player
        while not currentNode.isLeaf(currentPlayer):
            epsilon = self.mcts.args.epsilon
            edges = [edge for edge in currentNode.edges if edge.player == currentPlayer]
            if currentNode.isRootNode() and epsilon > 0:
                noise = np.random.dirichlet(np.full(len(edges), self.mcts.args.dirAlpha))
            else:
                epsilon = 0
                noise = np.zeros(len(edges))
            ns = sum(edge.N for edge in edges)
            if not epsilon:
                ns += EPS
            Us = np.full(len(edges), -math.inf)
            for idx, edge in enumerate(edges):
                if epsilon:
                    p = ((1 - epsilon) * edge.P + epsilon * noise[idx])
                else:
                    p = edge.P
                U = self.mcts.args.cpuct * \
                    p * \
                    np.sqrt(ns) / (1 + edge.N)
                score = edge.Q + U
                Us[idx] = score

            best_act = np.random.choice(np.flatnonzero(Us == Us.max()))
            selectedEdge = edges[best_act]
            selectedEdge.N += 1
            currentNode = selectedEdge.outNode
            backfillEdges.append(selectedEdge)
            currentPlayer *= -1
        value = currentNode.solveLeaf(currentPlayer) * currentPlayer * player
        for edge in backfillEdges:
            edge.W += value * edge.player
            edge.Q = edge.W / edge.N


class Edge:

    def __init__(self, inNode, outNode, player, prior, action):
        self.inNode = inNode
        self.outNode = outNode
        self.action = action
        self.player = player
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior


class MCTS2:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.tree = {}
        self.root = None
        self.Es = {}

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        currentPlayer = 1
        #totalTime = 0
        for i in range(self.args.numMCTSSims):
            #start = time.time()
            self.search(canonicalBoard)
            #end = time.time()
            #totalTime += (end - start)
        # print("elapsedTime: " + str(totalTime / self.args.numMCTSSims))
        edges = [edge for edge in self.root.edges if edge.player == currentPlayer]

        edgesInformation = dict([(edge.action, edge.N) for edge in edges])
        counts = np.array([edgesInformation.get(i, 0) for i in range(self.game.getActionSize())])

        if temp == 0:
            probs = np.zeros(shape=(len(counts)), dtype=int)
            probs[np.random.choice(np.flatnonzero(counts == counts.max()))] = 1
            return probs

        if temp != 1:
            counts = np.power(counts, (1./temp))                    # [x**(1./temp) for x in counts]
        probs = np.divide(counts, np.sum(counts), dtype=np.single)  # [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, canonicalBoard):
        ident = self.calculateId(self.game, canonicalBoard)
        if ident in self.tree:
            currentNode = self.tree[ident]
        else:
            currentNode = Node(self, canonicalBoard, ident)
            self.addNode(currentNode, ident)
        self.root = currentNode
        currentNode.expand(1)
        return currentNode

    def addNode(self, node, identifier):
        self.tree[identifier] = node

    @staticmethod
    def calculateId(game, canonicalBoard):
        #return hashlib.md5((str(game.stringRepresentation(canonicalBoard))).encode('utf-8')).hexdigest()
        return game.stringRepresentation(canonicalBoard)
