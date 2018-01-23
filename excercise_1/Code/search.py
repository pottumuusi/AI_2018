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
    def __init__(self, problem):
        from util import Stack
        self.expandedCoordinates = {}
        self.unvisitedCoordinates = Stack()
        self.problem = problem

        self.DEBUG_STEP = False
        self.DEBUG_PRINTS = False

    def run(self):
        node = None

        while True:
            if self.DEBUG_STEP:
                raw_input()

            node = self.visitNext(node)

            if None == node:
                break

            if self.problem.isGoalState(node):
                if self.DEBUG_PRINTS:
                    print "Next node to visit is goal state. Node is: " + str(node)
                break

        # TODO return list of directions which lead to goal state
        return node

    # Update nodes which can be visited and give next node to visit
    def visitNext(self, node):
        if None == node:
            self.addUnvisitedNode(self.problem.getStartState())
        else:
            # DFS will always want to try and expand current node
            self.expand(node)

        # Visited nodes have now been updated if necessary. Data structure
        # containing unvisited nodes should now give correct next node on
        # next access.
        nextNode = self.pickNextNode()

        return nextNode

    # Add successors of node to structure of unvisited nodes
    def expand(self, node):
        if self.DEBUG_PRINTS:
            print "Expanding node: " + str(node)

        if self.isAlreadyExpanded(node):
            return

        # Avoid adding node under expansion to unvisited nodes. Change if
        # causes problems.
        self.markExpanded(node)

        successors = self.problem.getSuccessors(node)

        for s in successors:
            if self.isAlreadyExpanded(s):
                continue
            self.addUnvisitedNode(s)

        if self.DEBUG_PRINTS:
            print "Done expanding node: " + str(node)

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

    def markExpanded(self, node):
        if self.DEBUG_PRINTS:
            print "Consider following node as expanded: " + str(node)

        coordinates = self.extractCoordinates(node)
        self.expandedCoordinates[coordinates] = True

    def addUnvisitedNode(self, data):
        if self.DEBUG_PRINTS:
            print "Adding unvisited coordinates from data: " + str(data)

        coordinates = self.extractCoordinates(data)

        if self.DEBUG_PRINTS:
            print "Pushing coordinates" + str(coordinates)

        self.unvisitedCoordinates.push(coordinates)

    def pickNextNode(self):
        if self.DEBUG_PRINTS:
            print "Picking next node from unvisitedCoordinates: " + str(self.unvisitedCoordinates)

        if not self.unvisitedCoordinates.isEmpty():
            return self.unvisitedCoordinates.pop()

        return None

    # Need to handle following different cases:
    # * getStartState() returns a tuple with coordinates.
    # * getSuccessors() returns a tuple which contains:
    #       1. tuple with coordinates
    #       2. Direction
    #       3. Cost(?)
    def extractCoordinates(self, data):
        if tuple is type(data[0]):
            return data[0]
        elif tuple is type(data):
            return data
        else:
            raise Exception("Failed to extract coordinates from data: " + str(data))

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

    search = Search(problem)
    startTime = time.time()
    result = search.run()
    stopTime = time.time()

    searchFinishBanner = "\n"
    searchFinishBanner += "==================== Search Done ====================" + "\n"
    searchFinishBanner += "Got search result: " + str(result) + "\n"
    searchFinishBanner += "Result is goal state? -> " + str(problem.isGoalState(result)) + "\n"
    searchFinishBanner += "Search took %s seconds" % (stopTime - startTime) + "\n"
    searchFinishBanner += "==================== Search Done ====================" + "\n"

    print searchFinishBanner

    latestRunFile = open('latest_run.txt', 'w')
    latestRunFile.write(searchFinishBanner)

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
