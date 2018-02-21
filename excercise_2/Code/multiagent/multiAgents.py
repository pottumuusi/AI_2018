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

GLOBAL_DEBUG = False

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
    
    if GLOBAL_DEBUG:
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

        self.PACMAN_LAYER_TURN = 0
        self.GHOST_LAYER_TURN = 1
        self.layerTurn = self.PACMAN_LAYER_TURN

        self.MAX_TAG = "max"
        self.MIN_TAG = "min"

        self.DEBUG_PRINTS = True

        self.searchTree = Stack()

        # self.alreadyDiscovered = {}
        self.parents = {}
        self.scores = {}
        self.selectedDirections = {}

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
            print "self.depth is: " + str(self.depth)
            print "gameState is: " + str(gameState)
            print "evaluationFunction is: " + str(self.evaluationFunction)
            print "ghostAmount is: " + str(self.getGhostAmount())
            try:
                print "rootGameState getScore gives: " + str(self.rootGameState.getScore())
            except:
                print "rootGameState getScore resulted in exception"

        # searchTree = self.constructSearchTree()
        self.constructSearchTree()

        if self.DEBUG_PRINTS:
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

            # print "printing stack again"
            # while not self.searchTree.isEmpty():
            #     print str(self.searchTree.pop())
            # print "done printing stack again"

        minimaxDecision = self.calculateMinimaxValue()
        # if self.DEBUG_PRINTS:
        #     print "minimaxValue is: " + str(minimaxValue)

        scoreKeys = self.scores.keys()
        selectedDirectionKeys = self.selectedDirections.keys()
        nextAction = self.optimalNextAction()

        if self.DEBUG_PRINTS:
            print "scoreKeys are: " + str(scoreKeys)

            for key in scoreKeys:
                print "For state " + str(key) + ", score is: " + str(self.scores[key])

            for key in selectedDirectionKeys:
                print "For state " + str(key) + ", direction is: " + str(self.selectedDirections[key])

            print "nextAction is: " + str(nextAction)

        # util.raiseNotDefined()

        # TODO replace with sane value when search working
        # return self.rootGameState.getLegalActions(self.PACMAN_INDEX)[2]

        return nextAction

    def optimalNextAction(self):
        return self.selectedDirections[self.rootGameState]

    def constructSearchTree(self):
        maxDepth = self.getDepth()
        depth = 1

        if self.DEBUG_PRINTS:
            print "constructSearchTree, depth is: " + str(depth)

        nextRoundStates = []
        nextRoundStates.append(self.rootGameState)

        # print "ghostActions is: " + str(ghostActions)
        # print "pacmanActions is: " + str(pacmanActions)

        # print "pacmanPossibleStates is: " + str(pacmanPossibleStates)
        # print "pacmanStateScores is: " + str(pacmanStateScores)
        # print "pacmanStateScorePairs is: " + str(pacmanStateScorePairs)

        while depth <= maxDepth:
            if self.DEBUG_PRINTS:
                print "\n===== Creating layers on depth: " + str(depth) + " ======"
                print "nextRoundStates to pass are: " + str(nextRoundStates)

            nextRoundStates = self.createLayersForRound(nextRoundStates)
            # nextRoundStates = self.createAlternatingTreeLayerForRound(nextRoundStates)

            # pacmanActions = self.getLegalPacmanActions(gameState)
            # pacmanPossibleStates = self.getPossibleNextStates(gameState, pacmanActions, self.PACMAN_INDEX)

            depth = depth + 1 # TODO put to correct place

        if self.DEBUG_PRINTS:
            print "===== Max depth reached ====="


#            # TODO need to assign parents for states added to tree
#            for pacState in pacmanPossibleStates:
#                # tree.push(pacState)
#                ghostActions = self.getLegalGhostActions(pacState, ghostAmount)
#
#                # TODO Make new layer in tree for each ghost
#                for ghostIndex in range(0, len(ghostActions)):
#                    singleGhostActions = ghostActions[ghostIndex]
#                    ghostStates = self.getPossibleNextStates(gameState, singleGhostActions, ghostIndex + 1)
#                    print "for ghost [" + str(ghostIndex) + "] got ghostStatuses: " + str(ghostStates)

    # Alternate between creating (pacman / max) and (ghost / min) layers on
    # calls.
    # def createAlternatingTreeLayerForRound(self, roundStates):
    #     ghostAmount = self.getGhostAmount()
    #     nextRoundStates = []
    #     newStates = [] 
    #     if self.DEBUG_PRINTS:
    #         print "createAlternatingLayerForRound, roundStates is: " + str(roundStates)

    #     if self.layerTurn == self.PACMAN_LAYER_TURN:
    #         if self.DEBUG_PRINTS:
    #             print "Creating pacman layer"
    #         for state in roundStates:
    #             if self.DEBUG_PRINTS:
    #                 print "Handling state " + str(state) + " of roundStates"
    #             # newStates = newStates + self.createTreeLayerForAgent(state, self.PACMAN_INDEX)
    #             nextRoundStates = nextRoundStates + self.createTreeLayerForAgent(state, self.PACMAN_INDEX)
    #         if self.DEBUG_PRINTS:
    #             print "Done creating pacman layer"

    #         self.layerTurn = self.GHOST_LAYER_TURN
    #     else:
    #         if self.DEBUG_PRINTS:
    #             print "Creating ghost layers"
    #         for ghostNum in range(0, ghostAmount):
    #             print "Creating layer for ghostNum: " + str(ghostNum)
    #             # for state in newStates:
    #             for state in roundStates:
    #                 nextRoundStates = nextRoundStates + self.createTreeLayerForAgent(state, ghostNum + 1)
    #         if self.DEBUG_PRINTS:
    #             print "Done creating ghost layers"

    #         self.layerTurn = self.PACMAN_LAYER_TURN

    #     return nextRoundStates

    def createLayersForRound(self, roundStates):
        ghostAmount = self.getGhostAmount()
        nextRoundStates = []
        newStates = []

        if self.DEBUG_PRINTS:
            print "createLayersForRound, roundStates is: " + str(roundStates)

        if self.DEBUG_PRINTS:
            print "Creating pacman layer"
        for state in roundStates:
            if self.DEBUG_PRINTS:
                print "Handling state " + str(state) + " of roundStates"
            newStates = newStates + self.createTreeLayerForAgent(state, self.PACMAN_INDEX)
        if self.DEBUG_PRINTS:
            print "Done creating pacman layer"

        if [] == newStates and self.DEBUG_PRINTS:
            if self.DEBUG_PRINTS:
                print "===== Supposedly was layer with leaf nodes. Ending early ====="
            return newStates

        if self.DEBUG_PRINTS:
            print "Creating ghost layers"
        for ghostNum in range(0, ghostAmount):
            print "Creating layer for ghostNum: " + str(ghostNum)
            for state in newStates:
                nextRoundStates = nextRoundStates + self.createTreeLayerForAgent(state, ghostNum + 1)
        if self.DEBUG_PRINTS:
            print "Done creating ghost layers"
            if [] == nextRoundStates and self.DEBUG_PRINTS:
                print "===== Supposedly was layer with leaf nodes ====="

            # try:
            #     # It looks like states which have pacman in same position as
            #     # before can have different hash. Keeping track of already
            #     # discovered states when creating the tree seems to be the
            #     # wrong thing to do.
            #     if True == self.alreadyDiscovered[state]:
            #         if self.DEBUG_PRINTS:
            #             print "Already discovered"
            #         pass
            # except KeyError:
            #     self.alreadyDiscovered[state] = True
            #     # newStates + self.createTreeLayerForAgent(state, self.PACMAN_INDEX)

            # newStates = newStates + self.createTreeLayerForAgent(state, self.PACMAN_INDEX)
            # for newState in newStates:
            #     try:
            #         if True == self.alreadyDiscovered[newState]:
            #             if self.DEBUG_PRINTS:
            #                 print "Already discovered"
            #             pass
            #     except KeyError:
            #         self.alreadyDiscovered[newState] = True
            #         newStates + self.createTreeLayerForAgent(state, self.PACMAN_INDEX)
            #         # nextRoundStates.append(newState)


        # Return states which are possible next pacman states. They will be
        # used next round to get the next possible moves.
        # When previously returned ghost positions which were used to get
        # pacman positions, empty actions were returned by getLegalActions.
        # Makes sense as these possible moves for ghosts are far from pacman.
        # Moving to them would be teleporting.
        # return newStates

        return nextRoundStates

    def createTreeLayerForAgent(self, gameState, agentIndex):
        if self.DEBUG_PRINTS:
            print "Creating tree layer for agent with " \
                + "gameState: " + str(gameState) \
                + ", agentIndex: " + str(agentIndex)

        actions = self.getLegalAgentActions(gameState, agentIndex)
        possibleNextStates = self.getPossibleNextStates(gameState, actions, agentIndex)

        for state in possibleNextStates:
            if self.PACMAN_INDEX == agentIndex:
                tag = self.MAX_TAG
            else:
                tag = self.MIN_TAG

            self.searchTree.push(self.makeSearchNode(state, tag))

        return possibleNextStates

    def makeSearchNode(self, state, tag):
        return (state, tag)

    def calculateMinimaxValue(self):
        # Stack is ordered so that states of last layer are popped first,
        # then states of second last layer etc.

        while not self.searchTree.isEmpty():
            searchNode = self.searchTree.pop()
            state = searchNode[self.SEARCH_STATE_POS]

            if self.MAX_TAG == searchNode[self.SEARCH_MINMAX_POS]:
                maxTurn = True
            elif self.MIN_TAG == searchNode[self.SEARCH_MINMAX_POS]:
                maxTurn = False
            else:
                raise Exception("Unrecognized search node minmax pos value")

            if self.DEBUG_PRINTS:
                print "In Minimax value calculation handling next state: " + str(state)

            parent = self.parents[state]

            # Will cause exception if not leaf node
            try:
                stateScore = self.evaluationFunction(state)
            except Exception:
                stateScore = self.scores[state]

            try:
                parentScore = self.scores[parent[self.PARENT_STATE_POS]]
                if maxTurn and (stateScore > parentScore):
                    self.scores[parent[self.PARENT_STATE_POS]] = stateScore
                    self.selectedDirections[parent[self.PARENT_STATE_POS]] = parent[self.PARENT_ACTION_POS]
                    if self.DEBUG_PRINTS:
                        print "For state " + str(parent[self.PARENT_STATE_POS]) \
                                + "\n\tset score: " + str(stateScore) \
                                + "\n\tset selectedDirection: " + str(parent[self.PARENT_ACTION_POS])
                elif (not maxTurn) and (stateScore < parentScore):
                    self.scores[parent[self.PARENT_STATE_POS]] = stateScore
                    self.selectedDirections[parent[self.PARENT_STATE_POS]] = parent[self.PARENT_ACTION_POS]
                    if self.DEBUG_PRINTS:
                        print "For state " + str(parent[self.PARENT_STATE_POS]) \
                                + "\n\tset score: " + str(stateScore) \
                                + "\n\tset selectedDirection: " + str(parent[self.PARENT_ACTION_POS])
                else:
                    if self.DEBUG_PRINTS:
                        print "\tmaxTurn: " + str(maxTurn) + ", stateScore " + str(stateScore) + " not change in parent score " + str(parentScore)
            except KeyError:
                # Give current value to parent in case none is present.
                self.scores[parent[self.PARENT_STATE_POS]] = stateScore
                self.selectedDirections[parent[self.PARENT_STATE_POS]] = parent[self.PARENT_ACTION_POS]
                if self.DEBUG_PRINTS:
                    print "For state " + str(parent[self.PARENT_STATE_POS]) \
                            + "\n\t set score: " + str(stateScore) \
                            + "\n\t set selectedDirection: " + str(parent[self.PARENT_ACTION_POS])

        # self.searchTree
        # Score is calculated the first time when max depth is found

        # pacmanStateScores = self.getStateScores(pacmanPossibleStates)
        # pacmanStateScorePairs = self.makeScorePairs(pacmanPossibleStates, pacmanStateScores)

        return "Minimax calculation not yet implemented"

    def makeScorePairs(self, states, scores):
        pairs = []

        if len(states) != len(scores):
            raise Exception("Need equal amount of states and scores to make pairs")

        for i in range(0, len(states)):
            pair = ()
            pair[self.PAIR_STATE_INDEX] = states[i]
            pair[self.PAIR_SCORE_INDEX] = score[i]
            pairs.append(pair)

        return pairs

    def getLegalGhostActions(self, gameState, ghostAmount):
        allLegalActions = []

        if self.DEBUG_PRINTS:
            print "getting legal ghost actions, ghost amount is: " + str(ghostAmount)

        for i in range(self.FIRST_GHOST_INDEX, self.FIRST_GHOST_INDEX + ghostAmount):
            legalActions = gameState.getLegalActions(i)
            allLegalActions.append(legalActions)

        gameState.getLegalActions()

        return allLegalActions

    # def getPossibleNextGhostStates(self, actions, gameState):
    #     nextStatuses
    #     return

    def getLegalPacmanActions(self, gameState):
        return gameState.getLegalActions(self.PACMAN_INDEX)

    def getLegalAgentActions(self, gameState, index):
        actions = gameState.getLegalActions(index)

        if self.DEBUG_PRINTS:
            print "Got actions: " + str(actions)

        return actions

    def getPossibleNextStates(self, gameState, actions, index):
        nextStates = []

        for a in actions:
            state = gameState.generateSuccessor(index, a)
            self.parents[state] = (gameState, a)
            print "For action " + str(a) + " got successor: " + str(state)
            nextStates.append(state)

        return nextStates

    def getStateScores(self, states):
        scores = []

        for s in states:
            score = self.evaluationFunction(s)
            scores.append(score)

        return scores

    def getGhostAmount(self):
        return self.rootGameState.getNumAgents() - 1

    def getDepth(self):
        return self.depth

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

