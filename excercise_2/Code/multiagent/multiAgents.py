# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

GLOBAL_DEBUG_PRINTS = False

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.DEBUG_PRINTS = False
        self.DEBUG_STEP = False
        self.RESULT_PRINTS = False

        self.PLAYER_POSITION_KEY = "player_position"
        self.GHOST_POSITIONS_KEY = "ghost_positions"
        self.FOOD_POSITIONS_KEY = "food_positions"
        self.ACTION_KEY = "action"

        self.GHOST_DISTANCE_WEIGHT = 0.15
        self.FOOD_DISTANCE_WEIGHT = 0.85

        self.MAX_SINGLE_BASE_SCORE = pow(1, -1) * 10
        self.WORST_SCORE_MULT = self.MAX_SINGLE_BASE_SCORE * 4
        self.SECOND_WORST_SCORE_MULT = self.MAX_SINGLE_BASE_SCORE * 2
        self.THIRD_WORST_SCORE_MULT = self.MAX_SINGLE_BASE_SCORE * 0.2

        self.currentGameState = None
        self.successorGameState = None
        self.remainingFood = None

        if self.DEBUG_PRINTS:
            print "self.WORST_SCORE_MULT is:" + str(self.WORST_SCORE_MULT)
            print "self.SECOND_WORST_SCORE_MULT is:" + str(self.SECOND_WORST_SCORE_MULT)
            print "self.THIRD_WORST_SCORE_MULT is:" + str(self.THIRD_WORST_SCORE_MULT)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        if self.RESULT_PRINTS:
            print ""
            print "=== Direction picked ==="
            print ""

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Original usage examples
        #
        # Useful information you can extract from a GameState (pacman.py)
        # successorGameState = currentGameState.generatePacmanSuccessor(action)
        # newPos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # score = successorGameState.getScore()
        #
        # Original usage examples

        "*** YOUR CODE HERE ***"

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        self.currentGameState = currentGameState
        self.successorGameState = successorGameState

        availableFood = self.currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        ghostPositions = currentGameState.getGhostPositions()
        foodPositions = self.extractFoodPositions(availableFood, newPos)

        stateInformation = {
            self.PLAYER_POSITION_KEY : newPos,
            self.GHOST_POSITIONS_KEY : ghostPositions,
            self.FOOD_POSITIONS_KEY : foodPositions,
            self.ACTION_KEY : action
        }

        score = self.calculateScore(stateInformation)

        if self.RESULT_PRINTS:
            print "=== printing evaluation variables ==="
            print "action: " + str(action)
            print "newPos: " + str(newPos)
            print "score: " + str(score)
            print "=== done printing evaluation variables ==="

        if self.DEBUG_PRINTS:
            print "successorGameState: " + str(successorGameState)
            print "newFood: " + str(newFood)
            print "newGhostStates: " + str(newGhostStates)
            print "newScaredTimes: " + str(newScaredTimes)

        if self.DEBUG_STEP:
            raw_input()

        return score

    def calculateScore(self, info):
        distancesToGhosts = []
        distancesToFood = []
        playerPos = info[self.PLAYER_POSITION_KEY]
        ghostPositions = info[self.GHOST_POSITIONS_KEY]
        foodPositions = info[self.FOOD_POSITIONS_KEY]
        action = info[self.ACTION_KEY]

        if self.DEBUG_PRINTS:
            print "ghostPositions: " + str(ghostPositions)
            print "foodPositions: " + str(foodPositions)

        distancesToFood = self.fillDistances(playerPos, foodPositions)
        distancesToGhosts = self.fillDistances(playerPos, ghostPositions)
        score = self.calculateScoreFromDistances(distancesToFood, distancesToGhosts)

        return score

    def calculateScoreFromDistances(self, distancesToFood, distancesToGhosts):
        foodScore = 0
        ghostScore = 0

        if self.DEBUG_PRINTS:
            self.distanceDebugPrints()

        for dist in distancesToFood:
            calculatedScore = self.calculateFoodScore(dist)
            foodScore = foodScore + calculatedScore
            if self.DEBUG_PRINTS:
                foodScoreDebugPrint(foodScore, dist, calculatedScore)

        for dist in distancesToGhosts:
            ghostScore = ghostScore + self.calculateGhostScore(dist)

        foodScore = foodScore * self.FOOD_DISTANCE_WEIGHT
        ghostScore = ghostScore * self.GHOST_DISTANCE_WEIGHT
        totalScore = foodScore + ghostScore

        if self.RESULT_PRINTS:
            print "=== printing evaluation variables ==="
            print "Food score: " + str(foodScore)
            print "Ghost score: " + str(ghostScore)
            print "=== done printing evaluation variables ==="

        return totalScore

    def calculateFoodScore(self, distToFood):
        score = self.baseScoreFromDistance(distToFood)

        # The less food the more it weights in decision making
        # Supposedly this still rarely causes pacman to die
        score = score / self.getRemainingFood()
        score = score * 100

        return score

    def calculateGhostScore(self, distToGhost):
        if self.DEBUG_PRINTS:
            print "calculateGhostScore, distToGhost is: " + str(distToGhost)

        # Calculate positive values when quite far from ghosts.
        if distToGhost > 2:
            score = distToGhost
            score = score * self.getRemainingFood() # Stay farther from ghosts when more food left
            score = score / 100
            return score

        score = self.baseScoreFromDistance(distToGhost)

        if 0 == distToGhost:
            score = score * self.WORST_SCORE_MULT
        elif 1 == distToGhost:
            score = score * self.SECOND_WORST_SCORE_MULT
        elif 2 == distToGhost:
            score = score * self.THIRD_WORST_SCORE_MULT

        score = score * 100
        score = score * -1

        return score

    def baseScoreFromDistance(self, distance):
        score = None

        if 0 == distance:
            score = self.MAX_SINGLE_BASE_SCORE
        else:
            score = pow(distance, -1) # Closer distance gives exponentially larger score

        return score

    def fillDistances(self, playerPos, positions):
        distances = []

        for pos in positions:
            dist = manhattanDistance(playerPos, pos)
            if self.DEBUG_PRINTS:
                print "Distance from " + str(playerPos) + " to " + str(pos) + " is: " + str(dist)
            distances.append(dist)

        return distances

    def extractFoodPositions(self, food, newPos):
        from game import Grid

        foodPositions = []

        if not isinstance(food, Grid):
            raise Exception("Trying to extract food positions from a non-Grid object")

        for y in range(0, food.getHeight()):
            for x in range(0, food.getWidth()):
                if (True == food[x][y]):
                    foodPositions.append((x, y))

        self.setRemainingFood(len(foodPositions))

        return foodPositions

    def setRemainingFood(self, remaining):
        self.remainingFood = remaining

    def getRemainingFood(self):
        return self.remainingFood

    def foodScoreDebugPrint(self, foodScore, dist, calculatedScore):
        print "foodScore: " + str(foodScore) + " dist: " + str(dist) \
                + " calculatedScore: " + str(calculatedScore)

    def distanceDebugPrints(self, distancesToGhosts, distancesToFood):
        print "printing distances to ghosts:"
        for dist in distancesToGhosts:
            print str(dist)
        print "done printing distances to ghosts"

        print "printing distances to food:"
        for dist in distancesToFood:
            print str(dist)
        print "done printing distances to food"

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    if GLOBAL_DEBUG_PRINTS:
        print "scoreEvaluation function running"

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        from util import Stack

        self.DEBUG_PRINTS = False

        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        self.rootGameState = None
        self.PACMAN_INDEX = 0
        self.FIRST_GHOST_INDEX = 1

        self.PAIR_STATE_INDEX = 0
        self.PAIR_SCORE_INDEX = 1

        self.PARENT_STATE_POS = 0
        self.PARENT_ACTION_POS = 1

        self.SEARCH_STATE_POS = 0
        self.SEARCH_MINMAX_POS = 1
        self.SEARCH_ACTION_POS = 2

        self.PACMAN_LAYER_TURN = 0
        self.GHOST_LAYER_TURN = 1
        self.layerTurn = self.PACMAN_LAYER_TURN

        self.MAX_TAG = "max"
        self.MIN_TAG = "min"

        self.layer = 0

        self.searchTree = Stack()

        self.parents = {}
        self.scores = {}
        self.selectedDirections = {}
        self.layers = {}
        self.leafStates = {}

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def initializeMemberStructures(self):
        from util import Stack

        self.layers = {}
        self.leafStates = {}

        self.searchTree = Stack()

        self.parents = {}
        self.scores = {}
        self.selectedDirections = {}

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        self.rootGameState = gameState

        if self.DEBUG_PRINTS:
            self.printSearchStartStatus()

        self.initializeMemberStructures()
        self.constructSearchTree()

        if self.DEBUG_PRINTS:
            self.printTreeContent()

        self.setStateTransitionsByMinimax()

        if self.DEBUG_PRINTS:
            self.printSearchEndStatus()

        nextAction = self.optimalNextAction()

        return nextAction

    def constructSearchTree(self):
        maxDepth = self.getDepth()
        depth = 1

        nextRoundStates = []
        nextRoundStates.append(self.rootGameState)

        while depth <= maxDepth:
            if self.DEBUG_PRINTS:
                print "\n===== Creating layers on depth: " + str(depth) + " ======"
                print "nextRoundStates to pass are: " + str(nextRoundStates)

            nextRoundStates = self.createLayersForRound(nextRoundStates)

            depth = depth + 1

        if self.DEBUG_PRINTS:
            print "===== Max depth reached ====="

    # Create search tree layer for pacman and for each ghost
    def createLayersForRound(self, roundStates):
        ghostAmount = self.getGhostAmount()
        nextRoundStates = []
        newStates = []

        newStates = self.createLayer(roundStates, self.PACMAN_INDEX)

        if [] == newStates:
            if self.DEBUG_PRINTS:
                print "===== Supposedly was layer with leaf nodes. Ending early ====="
            return newStates

        for ghostNum in range(0, ghostAmount):
            newStates = self.createLayer(newStates, ghostNum + 1)

        if [] == newStates:
            if self.DEBUG_PRINTS:
                print "===== Supposedly was layer with leaf nodes ====="

        # Return states which are possible next pacman states. They will be
        # used next round to get the next possible moves.
        return newStates

    def createLayer(self, layerStates, agentIndex):
        newStates = []

        if self.DEBUG_PRINTS:
            agentName = self.agentIndexName(agentIndex)
            print "Creating " + agentName + " layer. Layer: " + str(self.layer)

        for state in layerStates:
            if self.DEBUG_PRINTS:
                print "Handling state " + str((state, "debug_tuple")) + " of layerStates"
            newStates = newStates + self.createTreeLayerForAgent(state, agentIndex)

        if self.DEBUG_PRINTS:
            print "Done creating " + agentName + " layer. Layer: " + str(self.layer)

        self.layer = self.layer + 1

        return newStates

    def createTreeLayerForAgent(self, gameState, agentIndex):
        if self.DEBUG_PRINTS:
            print "Creating tree layer for agent with " \
                + "gameState: " + str((gameState, "debug_tuple")) \
                + ", agentIndex: " + str(agentIndex)

        actions = self.getLegalAgentActions(gameState, agentIndex)
        possibleNextStates = self.getPossibleNextStates(gameState, actions, agentIndex)

        if (len(actions) != len(possibleNextStates)):
            raise Exception("Different number of actions and possibleNextStates")

        for i in range(0, len(possibleNextStates)):
            if self.PACMAN_INDEX == agentIndex:
                tag = self.MAX_TAG
            else:
                tag = self.MIN_TAG

            state = possibleNextStates[i]
            action = actions[i]

            self.layers[state] = self.layer
            self.searchTree.push(self.makeSearchNode(state, tag, action, self.layer, gameState))

        return possibleNextStates

    def getLegalAgentActions(self, gameState, index):
        actions = gameState.getLegalActions(index)

        if self.DEBUG_PRINTS:
            print "Got actions: " + str(actions)

        return actions

    def getPossibleNextStates(self, gameState, actions, index):
        nextStates = []

        for a in actions:
            state = gameState.generateSuccessor(index, a)
            self.leafUpdate(gameState, state)
            self.addParent(state, gameState, a)
            nextStates.append(state)

            if self.DEBUG_PRINTS:
                print "From state: " + str((gameState, "debug_tuple")) \
                        + ", For action " + str(a) + " got successor: " \
                        + str((state, "debug_tuple"))

        return nextStates

    def leafUpdate(self, oldState, newState):
        if self.DEBUG_PRINTS:
            print "UNsetting leaf state: " + str((oldState, "debug_tuple"))
        self.leafStates[oldState] = False

        if self.DEBUG_PRINTS:
            print "setting leaf state: " + str((newState, "debug_tuple"))
        self.leafStates[newState] = True

    # Go through all states. Use value of each state to update value of parent
    # state if appropriate.
    def setStateTransitionsByMinimax(self):
        # Stack is ordered so that states of last layer are popped first,
        # then states of second last layer etc.

        while not self.searchTree.isEmpty():
            searchNode = self.searchTree.pop()
            state = searchNode[self.SEARCH_STATE_POS]
            action = searchNode[self.SEARCH_ACTION_POS]

            parent = self.getParent(state, action)

            maxTurn = self.nodeFromMaxLayer(searchNode)

            if self.DEBUG_PRINTS:
                print "In Minimax value calculation handling next state: " \
                        + str((state, "debug_tuple")) \
                        + ", parent is: " + str((parent, "debug_tuple")) \

            stateScore = self.getScoreOfState(state)

            try:
                self.parentScoreUpdateMinimax(parent, stateScore, maxTurn)
            except KeyError:
                self.setFirstParentScore(parent, stateScore, maxTurn)

    def parentScoreUpdateMinimax(self, parent, scoreCandidate, maxTurn):
        parentState = parent[self.PARENT_STATE_POS]
        actionToChild = parent[self.PARENT_ACTION_POS]
        parentScore = self.scores[parentState]

        parentLayer = self.getParentLayer(parentState)

        if maxTurn and (scoreCandidate > parentScore):
            self.scores[parentState] = scoreCandidate
            self.selectedDirections[parentState] = actionToChild
            if self.DEBUG_PRINTS:
                print "For state " + str((parentState, "debug_tuple")) \
                        + "\n\tat layer: " + str(parentLayer) \
                        + "\n\tupdated score: " + str(scoreCandidate) \
                        + "\n\tset selectedDirection: " + str(actionToChild) \
                        + "\n\t maxTurn: " + str(maxTurn)
        elif (not maxTurn) and (scoreCandidate < parentScore):
            self.scores[parentState] = scoreCandidate
            self.selectedDirections[parentState] = actionToChild
            if self.DEBUG_PRINTS:
                print "For state " + str((parentState, "debug_tuple")) \
                        + "\n\tat layer: " + str(parentLayer) \
                        + "\n\tupdated score: " + str(scoreCandidate) \
                        + "\n\tset selectedDirection: " + str(actionToChild) \
                        + "\n\t maxTurn: " + str(maxTurn)
        else:
            if self.DEBUG_PRINTS:
                print "\tmaxTurn: " + str(maxTurn) + ", scoreCandidate " \
                        + str(scoreCandidate) \
                        + ", not change in parent score " + str(parentScore)

    def setFirstParentScore(self, parent, scoreCandidate, maxTurn):
        parentState = parent[self.PARENT_STATE_POS]
        actionToChild = parent[self.PARENT_ACTION_POS]

        parentLayer = self.getParentLayer(parentState)

        self.scores[parentState] = scoreCandidate
        self.selectedDirections[parentState] = actionToChild
        if self.DEBUG_PRINTS:
            print "For state " + str((parentState, "debug_tuple")) \
                    + "\n\t at layer: " + str(parentLayer) \
                    + "\n\t set score: " + str(scoreCandidate) \
                    + "\n\t set selectedDirection: " + str(actionToChild) \
                    + "\n\t maxTurn: " + str(maxTurn)

    def addParent(self, childState, parentState, action):
        if self.DEBUG_PRINTS:
            print "Setting parent: " + str((parentState, "debug_tuple")) \
                    + " to child: " + str((childState, "debug_tuple"))

        self.parents[(childState, action)] = (parentState, action)

    def makeSearchNode(self, state, tag, action, layer, gameState):
        return (state, tag, action, layer, gameState)

    def getParent(self, state, action):
        return self.parents[(state, action)]

    def getGhostAmount(self):
        return self.rootGameState.getNumAgents() - 1

    def getDepth(self):
        return self.depth

    def isLeafState(self, state):
        return self.leafStates[state]

    def optimalNextAction(self):
        return self.selectedDirections[self.rootGameState]

    def nodeFromMaxLayer(self, node):
        minMaxTag = node[self.SEARCH_MINMAX_POS]

        if self.MAX_TAG == minMaxTag:
            return True
        elif self.MIN_TAG == minMaxTag:
            return False
        else:
            raise Exception("Unrecognized search node minmax pos value")

    def getParentLayer(self, parentState):
        if self.rootGameState == parentState:
            parentLayer = "<not found>"
        else:
            parentLayer = self.layers[parentState]

    def getScoreOfState(self, state):
        stateLayer = self.layers[state]

        if self.isLeafState(state):
            if self.DEBUG_PRINTS:
                print "Evaluating state score on layer: " + str(stateLayer)
            stateScore = self.evaluationFunction(state)
        else:
            if self.DEBUG_PRINTS:
                print "Getting preset score for state " \
                        + str((state, "debug_tuple")) \
                        + ", stateLayer is: " + str(stateLayer)
            stateScore = self.scores[state]

        if self.DEBUG_PRINTS:
            print "Got stateScore: " + str(stateScore)

        return stateScore

    def agentIndexName(self, index):
        if (index == self.PACMAN_INDEX):
            return "pacman"

        if (index > self.PACMAN_INDEX):
            return "ghost"

        raise Exception("Unexpected agent index: " + str(index))

    def printTreeContent(self):
        from util import Stack
        restoreStack = Stack()

        print "printing Stack content"
        while not self.searchTree.isEmpty():
            val = self.searchTree.pop()
            restoreStack.push(val)
            print str(val)

        while not restoreStack.isEmpty():
            self.searchTree.push(restoreStack.pop())
        print "done printing Stack content"

    def printSearchStartStatus(self):
        print "\n===== Minimax search starting ====="
        print "self.depth is: " + str(self.depth)
        print "gameState is: " + str(self.rootGameState)
        print "evaluationFunction is: " + str(self.evaluationFunction)
        print "ghostAmount is: " + str(self.getGhostAmount())
        try:
            print "rootGameState getScore gives: " + str(self.rootGameState.getScore())
        except:
            print "rootGameState getScore resulted in exception"

    def printSearchEndStatus(self):
        scoreKeys = self.scores.keys()
        selectedDirectionKeys = self.selectedDirections.keys()
        layerKeys = self.layers.keys()
        nextAction = self.optimalNextAction()

        print "scoreKeys are: " + str(scoreKeys)

        # layerKeys should contain all states
        for key in layerKeys:
            try:
                score = self.scores[key]
            except KeyError:
                score = "<not found>"

            try:
                direction = self.selectedDirections[key]
            except KeyError:
                direction = "<not found>"

            try:
                parent = self.parents[key]
            except KeyError:
                parent = "<not found>"

            print "For state " + str((key, "debug_tuple"))
            print "\tscore is: " + str(score)
            print "\tdirection is: " + str(direction)
            print "\tparent is: " + str((parent, "debug_tuple"))
            print "\tlayer is: " + str(self.layers[key])

        print "nextAction is: " + str(nextAction)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

