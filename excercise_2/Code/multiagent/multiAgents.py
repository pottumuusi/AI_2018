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
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        self.gameState = None
        self.PACMAN_INDEX = 0
        self.FIRST_GHOST_INDEX = 1

        self.PAIR_STATE_INDEX = 0
        self.PAIR_SCORE_INDEX = 1

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

        print "self.depth is: " + str(self.depth)
        print "gameState is: " + str(gameState)
        self.gameState = gameState

        evalFunction = self.evaluationFunction
        ghostAmount = self.gameState.getNumAgents() - 1

        pacmanActions = self.getLegalPacmanActions()
        ghostActions = self.getLegalGhostActions(ghostAmount)


        pacmanPossibleStates = self.getPossibleNextPacmanStates(pacmanActions)

        # TODO Instead of getting scores while discovering states, the search
        # tree needs to be constructed first. Then scores are selected
        # starting from leaves and traversing tree to the direction of root.
        pacmanStateScores = self.getStateScores(pacmanPossibleStates)
        pacmanStateScorePairs = self.makeScorePairs(pacmanPossibleStates, pacmanStateScores)

        print "evalFunction is: " + str(evalFunction)
        print "ghostAmount is: " + str(ghostAmount)

        print "ghostActions is: " + str(ghostActions)
        print "pacmanActions is: " + str(pacmanActions)

        print "pacmanPossibleStates is: " + str(pacmanPossibleStates)
        print "pacmanStateScores is: " + str(pacmanStateScores)
        print "pacmanStateScorePairs is: " + str(pacmanStateScorePairs)

        util.raiseNotDefined()

    def getLegalGhostActions(self, ghostAmount):
        allLegalActions = []

        print "getting legal ghost actions, ghost amount is: " + str(ghostAmount)

        for i in range(self.FIRST_GHOST_INDEX, self.FIRST_GHOST_INDEX + ghostAmount):
            legalActions = self.gameState.getLegalActions(i)
            allLegalActions.append(legalActions)

        self.gameState.getLegalActions()

        return allLegalActions

    def getLegalPacmanActions(self):
        return self.gameState.getLegalActions(self.PACMAN_INDEX)

    def getPossibleNextPacmanStates(self, actions):
        nextStates = []

        for act in actions:
            state = self.gameState.generateSuccessor(self.PACMAN_INDEX, act)
            nextStates.append(state)

        return nextStates

    def getStateScores(self, states):
        scores = []

        for s in states:
            score = self.evaluationFunction(s)
            scores.append(score)

        return scores

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

