# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class Search:
    def __init__(self, problem, searchType, heuristic=None, options={}):
        from util import Stack
        from util import Queue
        from util import PriorityQueue
        from util import InformativePriorityQueue

        from game import Directions

        global OPT_KEY_MULTIPLE_GOALS
        global OPT_KEY_STR_GOAL

        self.DEBUG_STEP = False
        self.DEBUG_PRINTS = False
        self.DEBUG_ROUTE_PRINTS = False
        self.SUPPRESS_ERRORS = False

        self.COORDINATES_POSITION = 0
        self.DIRECTION_POSITION = 1
        self.COST_POSITION = 2
        self.DFS_TYPE = "dfs"
        self.BFS_TYPE = "bfs"
        self.UCS_TYPE = "ucs"
        self.ASTAR_TYPE = "astar"
        self.EMPTY_NODE = (None, None, None)

        self.problem = problem
        self.searchType = searchType
        self.heuristic = heuristic
        self.options = options

        self.parents = {}
        self.finalCoordinates = None
        self.expandedCoordinates = {}
        self.rawNodes = {}

        if self.isAstarSearch():
            if None == self.heuristic:
                raise Exception("No heuristic provided for A* search")

        if self.isDepthFirstSearch():
            self.unvisitedCoordinates = Stack()
        elif self.isBreadthFirstSearch():
            self.unvisitedCoordinates = Queue()
        elif self.isUniformCostSearch() or self.isAstarSearch():
            self.unvisitedCoordinates = InformativePriorityQueue()
        else:
            raise Exception("Unknown search type: " + searchType)

    def run(self):
        node = None
        latestCoordinates = None

        while True:
            if self.DEBUG_STEP:
                raw_input()

            node = self.visitNext(node)

            # All nodes processed
            if self.EMPTY_NODE == node:
                break

            latestCoordinates = self.extractCoordinates(node)
            isGoalState = self.isNodeGoalState(node)

            if isGoalState:
                break

        self.finalCoordinates = latestCoordinates
        return self.constructFinalRoute()

    # Update nodes which can be visited and give next node to visit
    def visitNext(self, node):
        if None == node:
            startState = self.problem.getStartState()
            self.addNodeParent(startState, None) # Starting node has no parent
            self.addUnvisitedNode(startState)
        else:
            # DFS will always want to try and expand current node
            self.expand(node)

        # Visited nodes have now been updated if necessary. Data structure
        # containing unvisited nodes should now give correct next node on
        # next access.
        nextCoordinates = self.pickNextCoordinates()
        nextNode = self.makeNode(nextCoordinates)

        return nextNode

    # Add successors of node to structure of unvisited nodes.
    # Link successors to node under expansion by adding current node as a
    # parent for all successors.
    def expand(self, node):
        if self.DEBUG_PRINTS:
            print "Expanding node: " + str(node)

        if self.isAlreadyExpanded(node):
            return

        # Avoid adding node under expansion to unvisited nodes. Change if
        # causes problems.
        self.markExpanded(node)

        self.handleFringe(node)

        if self.DEBUG_PRINTS:
            print "Done expanding node: " + str(node)

    def handleFringe(self, node):
        coordinates = self.extractCoordinates(node)

        try:
            successors = self.problem.getSuccessors(coordinates)
        except TypeError:
            # At least in some autograder cases a node was required instead of
            # coordinates to get successors.
            rawNode = self.getRawNode(coordinates)
            successors = self.problem.getSuccessors(rawNode)

        for s in successors:
            if self.isAlreadyExpanded(s):
                # No updates if already expanded.
                continue
            if self.isBreadthFirstSearch() and self.isAlreadyDiscovered(s):
                continue
            if self.isSearchWithUpdate() and self.isAlreadyDiscovered(s):
                # It is possible for uniform search to find a lower cost route
                # to a node.
                self.updateUnvisitedNode(s, coordinates)
                continue

            # Add parent before adding unvisited nodes as route might need
            # to be constructed. Route is needed at least for getting a cost
            # for route leading to successor being added.
            self.addNodeParent(s, coordinates)
            self.addUnvisitedNode(s)

    # Generate list of directions leading from starting position to passed
    # coordinates.
    def constructRoute(self, initialCoordinates):
        route = []
        nextCoordinates = initialCoordinates

        if self.DEBUG_PRINTS:
            print "Constructing route starting from: " + str(initialCoordinates)

        try:
            nextParent = self.parents[nextCoordinates]
        except KeyError:
            # No parent for given initial coordinates. Expect these
            # to be the coordinates of first node.
            return route

        while True:
            nextParent = self.parents[nextCoordinates]

            if self.isParentStartingPosition(nextParent):
                break

            nextDirection = self.extractDirection(nextParent)
            nextCoordinates = self.extractCoordinates(nextParent)
            self.parentSanityCheck(nextParent, nextDirection, nextCoordinates)

            route.append(nextDirection)

        # Flip route as it is currently from finish to start
        route.reverse()

        if self.DEBUG_ROUTE_PRINTS:
            self.routeDebugPrint(route)

        return route

    # Generate list of directions from starting position to finish.
    def constructFinalRoute(self):
        return self.constructRoute(self.finalCoordinates)

    def isAlreadyExpanded(self, node):
        try:
            coordinates = self.extractCoordinates(node)
            alreadyExpanded = self.expandedCoordinates[coordinates]
        except KeyError:
            if self.DEBUG_PRINTS:
                print "Index " + str(coordinates) + " not found from expandedCoordinates"
            alreadyExpanded = False

        if True != alreadyExpanded and False != alreadyExpanded:
            raise Exception("Unexpected value in alreadyExpanded: " + str(alreadyExpanded));

        return alreadyExpanded

    def isAlreadyDiscovered(self, node):
        alreadyDiscovered = None

        try:
            coordinates = self.extractCoordinates(node)
            self.rawNodes[coordinates]
            alreadyDiscovered = True
        except KeyError:
            if self.DEBUG_PRINTS:
                print "Index " + str(coordinates) + " not found from rawNodes"
            alreadyDiscovered = False

        if None == alreadyDiscovered:
            raise Exception("Failed to check if node is already discovered")

        return alreadyDiscovered

    def isNodeGoalState(self, node):
        goalState = False
        coordinates = self.extractCoordinates(node)

        try:
            goalState = self.problem.isGoalState(coordinates)
        except AttributeError:
            # At least in some autograder cases a node was required instead of
            # coordinates to check for goal state.
            rawNode = self.getRawNode(coordinates)
            if self.DEBUG_PRINTS:
                print "Querying goal state with raw node: " + str(rawNode)
            goalState = self.problem.isGoalState(rawNode)

        return goalState

    def markExpanded(self, node):
        if self.DEBUG_PRINTS:
            print "Consider following node as expanded: " + str(node)

        coordinates = self.extractCoordinates(node)
        self.expandedCoordinates[coordinates] = True

    # Test if route leading to passed node has lower cost if its parent had
    # the passed coordinates. Update node position in coordinates waiting to
    # be picked and node parent if cost is lower.
    def updateUnvisitedNode(self, data, parentCoordinates):
        if not (self.isUniformCostSearch() or self.isAstarSearch()):
            raise Exception("Unexpected search updating visited node")

        if self.DEBUG_PRINTS:
            print "Updating unvisited node: " + str(data)

        costs = []

        successorCoordinates = self.extractCoordinates(data)
        newDirection = self.extractDirection(data)
        route = self.constructRoute(parentCoordinates)
        route.append(newDirection)

        if self.DEBUG_PRINTS:
            print "Getting cost of actions for route: " + str(route)

        if self.isUniformCostSearch():
            costs.append(self.problem.getCostOfActions(route))
        elif self.isAstarSearch():
            costs = self.calculateAstarCost(successorCoordinates, data, route)

        for cost in costs:
            wasChanged = self.unvisitedCoordinates.update(successorCoordinates, cost)

            if wasChanged:
                self.addNodeParent(data, parentCoordinates)

    def addUnvisitedNode(self, data):
        if self.DEBUG_PRINTS:
            print "Adding unvisited coordinates from data: " + str(data)

        coordinates = self.extractCoordinates(data)

        if self.isUniformCostSearch():
            route = self.constructRoute(coordinates)
            cost = self.problem.getCostOfActions(route)

            if self.DEBUG_PRINTS:
                print "Pushing coordinates: " + str(coordinates) + " With cost: " + str(cost)

            self.unvisitedCoordinates.push(coordinates, cost)
        elif self.isAstarSearch():
            costs = self.calculateAstarCost(coordinates, data)

            if self.DEBUG_PRINTS:
                print "Pushing coordinates: " + str(coordinates) + " With costs: " + str(costs)

            for cost in costs:
                self.unvisitedCoordinates.push(coordinates, cost)
        else:
            if self.DEBUG_PRINTS:
                print "Pushing coordinates: " + str(coordinates)
            self.unvisitedCoordinates.push(coordinates)

        self.rawNodes[coordinates] = data

    # Autograder had strings as goal states. For example Manhattan
    # heuristic can't be used to calculate cost from strings.
    # Nodes contain a value as a third item. Expect that this item
    # is the number of hops needed to take to reach goal. In other
    # words the values are interesting to greedy search part of A*.
    def calculateAstarCost(self, coordinates, data, premadeRoute=None):
        try:
            costs = self.calculateAstarCostWithHeuristic(coordinates, premadeRoute)
            if self.DEBUG_PRINTS:
                print "Calculated A* costs: " + str(costs)
            return costs
        except TypeError:
            costs = self.astarCostFromPredefinedGreedyCost(data, premadeRoute)
            if self.DEBUG_PRINTS:
                print "Calculated A* costs [predefined]: " + str(costs)
            return costs

        raise Exception("Could not calculate A* cost for" \
                + " coordinates: " + str(coordinates) \
                + " data:" + str(data) \
                + " premadeRoute: " + str(premadeRoute))

    def astarCostFromPredefinedGreedyCost(self, data, premadeRoute=None):
        costs = []

        coordinates = self.extractCoordinates(data)

        if None != premadeRoute:
            route = premadeRoute
        else:
            route = self.constructRoute(coordinates)

        predefinedCost = self.extractCost(data)
        uniformCost = self.problem.getCostOfActions(route)

        if None == predefinedCost:
            # Handle first node
            predefinedCost = 0

        costs.append(predefinedCost + uniformCost)

        if self.DEBUG_PRINTS:
            print "Got predefinedCost: " + str(predefinedCost) + " uniformCost: " + str(uniformCost)

        self.astarCostSanityCheck(costs)

        return costs

    def calculateAstarCostWithHeuristic(self, coordinates, premadeRoute=None):
        costs = []

        if None != premadeRoute:
            route = premadeRoute
        else:
            route = self.constructRoute(coordinates)

        uniformCost = self.problem.getCostOfActions(route)
        heuristicResult = self.heuristic(coordinates, self.problem)

        if self.DEBUG_PRINTS:
            print "HeuristicResult is: " + str(heuristicResult)

        if self.problemHasMultipleGoals():
            for greedyCost in heuristicResult:
                costs.append(greedyCost + uniformCost)
        else:
            greedyCost = heuristicResult
            costs.append(uniformCost + greedyCost)

        if self.DEBUG_PRINTS:
            print "Got greedyCost: " + str(greedyCost) + " uniformCost: " + str(uniformCost)

        self.astarCostSanityCheck(costs)

        return costs

    def calculateAstarCostForMultipleGoals(self, coordinates):
        goals = self.problem.goals

        for goal in goals:
            heuristicResult = self

    def pickNextCoordinates(self):
        if self.DEBUG_PRINTS:
            print "<< Picking next coordinate from unvisitedCoordinates >>"

        if not self.unvisitedCoordinates.isEmpty():
            return self.unvisitedCoordinates.pop()

        return None

    def addNodeParent(self, successor, parentCoordinates):
        successorCoordinates = self.extractCoordinates(successor)
        successorDirection = self.extractDirection(successor)
        parent = self.makeParent(parentCoordinates, successorDirection)

        self.parents[successorCoordinates] = parent

        if self.DEBUG_PRINTS:
            print "Added parent: " + str(parent) + " for: " + str(successorCoordinates)

    def makeNode(self, coordinates):
        return (coordinates, None, None)

    def makeParent(self, parentCoordinates, successorDirection):
        return (parentCoordinates, successorDirection)

    # Need to handle following different cases:
    # * getStartState() returns a tuple with coordinates.
    # * getSuccessors() returns a tuple which contains:
    #       1. tuple with coordinates
    #       2. Direction
    #       3. Cost(?)
    def extractCoordinates(self, data):
        if self.hasCoordinate(data) or self.hasAutograderCoordinate(data):
            return data[self.COORDINATES_POSITION]

        if self.isBareCoordinate(data) or self.isBareAutograderCoordinate(data):
            return data

        raise Exception("Failed to extract coordinates from data: " + str(data))

    def extractDirection(self, data):
        def directionFromData(d):
            dataLen = len(d) # Trying to get length of int led to exception

            if self.dataHasDirection(d, dataLen) or self.dataHasAutograderDirection(d, dataLen):
                return data[self.DIRECTION_POSITION]

        try:
            return directionFromData(data)
        except TypeError as e:
            if not self.SUPPRESS_ERRORS:
                print "Could not get direction for node. Exception: " + str(e)

        if self.isBareCoordinate(data) or self.isBareAutograderCoordinate(data):
            return None # First state does not contain direction

        raise Exception("Failed to extract direction from data: " + str(data))

    def extractCost(self, data):
        if self.isBareAutograderCoordinate(data):
            return None # First state does not contain cost

        dataLen = len(data)
        if self.dataHasAutograderCost(data, dataLen):
            return data[self.COST_POSITION]

        raise Exception("Failed to extract cost from data: " + str(data))

    def getFinalCoordinates(self):
        return self.finalCoordinates

    def getRawNode(self, coordinates):
        return self.rawNodes[coordinates]

    def isParentStartingPosition(self, parent):
        return None == self.extractDirection(parent)

    def hasAutograderCoordinate(self, data):
        return tuple is type(data) and str is type(data[self.COORDINATES_POSITION])

    def hasCoordinate(self, data):
        return tuple is type(data[self.COORDINATES_POSITION])

    def isBareCoordinate(self, data):
        return tuple is type(data) and 2 == len(data)

    def isBareAutograderCoordinate(self, data):
        return str is type(data)

    def dataHasDirection(self, data, dataLen):
        return dataLen > 1 and self.hasCoordinate(data)

    def dataHasAutograderDirection(self, data, dataLen):
        return dataLen > 1 and self.hasAutograderCoordinate(data)

    def dataHasAutograderCost(self, data, dataLen):
        return dataLen > 2 and self.hasAutograderCoordinate(data)

    def isDepthFirstSearch(self):
        return self.DFS_TYPE == self.searchType

    def isBreadthFirstSearch(self):
        return self.BFS_TYPE == self.searchType

    def isUniformCostSearch(self):
        return self.UCS_TYPE == self.searchType

    def isAstarSearch(self):
        return self.ASTAR_TYPE == self.searchType

    def isSearchWithUpdate(self):
        return self.isUniformCostSearch() or self.isAstarSearch()

    def problemHasMultipleGoals(self):
        return self.getOptionValue(OPT_KEY_MULTIPLE_GOALS)

    def problemHasStrGoal(self):
        return self.getOptionValue(OPT_KEY_STR_GOAL)

    def getOptionValue(self, key):
        try:
            return self.options[key]
        except KeyError:
            return False

    def parentSanityCheck(self, parent, direction, coordinates):
        if None == direction:
            raise Exception("Could not extract direction from parent: " + str(parent))

        if None == coordinates:
            raise Exception("Could not extract coordinates from parent: " + str(parent))

    def astarCostSanityCheck(self, costs):
        if [] == costs:
            raise Exception("Failed to calculate A* cost from coordinates: " \
                    + str(coordinates) + " and premadeRoute: " \
                    + str(premadeRoute))

        if not isinstance(costs, list):
            raise Exception("Ended up with non-list costs: " + str(costs))

    def routeDebugPrint(self, route):
        print "---== Printing route ==---"
        print str(route)
        print "---== Done printing route ==---"

def getSearchFinishBanner(route, finalCoordinates, problem, searchTime):
    bannerDisabled = True
    isGoalState = None

    if bannerDisabled:
        return ""

    try:
        isGoalState = problem.isGoalState(finalCoordinates)
    except AttributeError:
        isGoalState = problem.isGoalState(search.getRawNode(finalCoordinates))

    searchFinishBanner = "\n"
    searchFinishBanner += "==================== Search Done ====================" + "\n"
    searchFinishBanner += "Got route: " + str(route) + "\n"
    searchFinishBanner += "Result is goal state? -> " + str(isGoalState) + "\n"
    searchFinishBanner += "Search took %s seconds" % (searchTime) + "\n"
    searchFinishBanner += "==================== Search Done ====================" + "\n"

    return searchFinishBanner

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    import time

    search = Search(problem, "dfs")
    startTime = time.time()
    route = search.run()
    stopTime = time.time()

    finalCoordinates = search.getFinalCoordinates()
    searchFinishBanner = getSearchFinishBanner(route, finalCoordinates, problem, stopTime - startTime)

    print searchFinishBanner
    latestRunFile = open('latest_run.txt', 'w')
    latestRunFile.write(searchFinishBanner)

    return route

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    import time

    search = Search(problem, "bfs")
    startTime = time.time()
    route = search.run()
    stopTime = time.time()

    finalCoordinates = search.getFinalCoordinates()
    searchFinishBanner = getSearchFinishBanner(route, finalCoordinates, problem, stopTime - startTime)

    print searchFinishBanner
    latestRunFile = open('latest_run.txt', 'w')
    latestRunFile.write(searchFinishBanner)

    return route

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    import time

    search = Search(problem, "ucs")
    startTime = time.time()
    route = search.run()
    stopTime = time.time()

    finalCoordinates = search.getFinalCoordinates()
    searchFinishBanner = getSearchFinishBanner(route, finalCoordinates, problem, stopTime - startTime)

    print searchFinishBanner
    latestRunFile = open('latest_run.txt', 'w')
    latestRunFile.write(searchFinishBanner)

    return route

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def isMultiGoalProblem(problem):
    try:
        goal = problem.goal
        return False
    except AttributeError:
        pass

    goals = problem.goals

    return True

def isStrGoalProblem(problem):
    if isMultiGoalProblem(problem):
        goal = problem.goals[0]
    else:
        goal = problem.goal

    return str is type(goal)

def aStarSearch(problem, heuristic=None):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    import time
    from searchAgents import manhattanHeuristic
    from searchAgents import manhattanHeuristicMultiGoal

    options = {}

    if None == heuristic and isMultiGoalProblem(problem):
        heuristic = manhattanHeuristicMultiGoal
        options[OPT_KEY_MULTIPLE_GOALS] = True
    elif None == heuristic:
        heuristic = manhattanHeuristic

    if isStrGoalProblem(problem):
        options[OPT_KEY_STR_GOAL] = True

    search = Search(problem, "astar", heuristic, options)

    startTime = time.time()
    route = search.run()
    stopTime = time.time()

    finalCoordinates = search.getFinalCoordinates()
    searchFinishBanner = getSearchFinishBanner(route, finalCoordinates, problem, stopTime - startTime)

    print searchFinishBanner
    latestRunFile = open('latest_run.txt', 'w')
    latestRunFile.write(searchFinishBanner)

    return route


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

OPT_KEY_MULTIPLE_GOALS = "multiple_goals"
OPT_KEY_STR_GOAL = "str_goal"
