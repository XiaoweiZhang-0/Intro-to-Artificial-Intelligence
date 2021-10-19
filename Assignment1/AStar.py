import MazeGenerator as MG
import random
import numpy as np
import BinHeap as BH
# 
# OpenList：list with binary heap
# Cell: current cell, includes coordinates, f_value, g_value
# Traversed_Route：list, mark the route from starting point to current cell
# Cal_Path: current calculated path from current cell to goal
# blockedCells: encountered blocked_cells with their coordinates
# 

# Helper Function
# findNeighbors: find all the neighbors, put blocked neighbors into blocked_cells. put unblocked_cells in the open_list
# H_value: return the H_value of the goal_cell
# find_route: find the route from current cell to the goal cell, returns a list = Cal_Path
#           input: blockedCells, curCell, goal
#           output: Calculated Path

# calculate Manhattan distance as h value
class Cell:
    def __init__(self, coord, fValue, gValue):
        self.coord = coord
        self.fValue = fValue
        self.gValue = gValue

def hValue(currCoord, goalCoord):
    return abs(currCoord[0] - goalCoord[0]) + abs(currCoord[1] - goalCoord[1])

def isValid(x, y):
    if(x >= 0 and x <= 100 and y >= 0 and y <= 100):
        return True
    else:
        return False
def isBlocked(maze, x, y):
    if(maze[x][y] == 0):
        return True
    else:
        return False
def findRoute(curCell, goal, openList, maze):
    localCell = curCell
    calPath = [localCell.coord]
    while localCell.coord != goal:
        findNeighbors(localCell, openList, goal, maze)
        localCell = BH.pop(openList)
        calPath.append(localCell.coord)
    return calPath
        # if isBlocked(maze, curCell.coord[0], curCell.coord[1]):
        #     return("No route found")
    

def findNeighbors(curCell, openList, goal, maze):
    curCell = curCell
    x = curCell.coord[0]
    y = curCell.coord[1]
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    for neighbor in neighbors:
        if isValid(neighbor[0], neighbor[1]):
            coord = neighbor
            # if(isBlocked(maze, coord[0], coord[1])):
            #     gVal = np.inf
            #     fVal = np.inf
            # else:
            gVal = curCell.gValue + 1
            fVal = gVal + hValue(coord, goal)
            newCell = Cell(coord, fVal, gVal)
            BH.insert(newCell, openList)

def aStar():
    # initialize 
    maze = MG.generateMaze()
    startCoord = (random.randint(0, 100), random.randint(0, 100))
    goal = (random.randint(0, 100), random.randint(0, 100))
    if maze[startCoord[0]][startCoord[1]] is 0 or maze[goal[0]][goal[1]] is 0:
        print("there is no path from startpoint to goal")
        return
    startHValue = hValue(startCoord, goal)
    startCell = Cell(startCoord, startHValue, 0)
    openList = [startCell]
    curCell = startCell
    travPath = [startCell.coord]

    while ()
    calPath = findRoute()

aStar()