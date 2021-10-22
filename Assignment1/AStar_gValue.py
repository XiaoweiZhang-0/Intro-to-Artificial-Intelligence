import MazeGenerator as MG
import random
import numpy as np
import BinHeap_large as BH_l
import BinHeap_small as BH_s
import time

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

num_rows = 101
num_cols = 101

class Cell:
    def __init__(self, coord, fValue, gValue):
        self.coord = coord
        self.fValue = fValue
        self.gValue = gValue

# calculate Manhattan distance as h value
def hValue(currCoord, goalCoord):
    return abs(currCoord[0] - goalCoord[0]) + abs(currCoord[1] - goalCoord[1])

def isValid(x, y):
    if(x >= 0 and x <= num_cols-1 and y >= 0 and y <= num_rows-1):
        return True
    else:
        return False

def isBlocked(blockedList, neighbor):
    # print('blocked list is',blockedList)
    # print('neighbor is ', neighbor)
    if(neighbor in blockedList):
        # print('this is blocked')
        return True
    else:
        return False

def findRoute_gs(goal, openList, closedList, search, counter, blockedList, gValue):
    # print('search is ',search)
    # print('-------------------')
    curCell = BH_s.pop(openList)
    path = {}
    if(not curCell):
        return False
    # print(curCell.coord)
    closedList.append(curCell.coord)
    while curCell.coord != goal:
        # closedList.append(curCell)
        findNeighbors_gs(curCell, openList, goal, search, counter, path, blockedList, closedList, gValue)
        curCell = BH_s.pop(openList)
        if(not curCell):
            return False
        closedList.append(curCell.coord)
        # print(curCell.coord)
    # print('end find route -----------------')
    return path
    
def findNeighbors_gs(curCell, openList, goal, search, counter, path, blockedList, closedList, gValue):
    x = curCell.coord[0]
    y = curCell.coord[1]
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    for neighbor in neighbors:
        if isValid(neighbor[0], neighbor[1]) and not isBlocked(blockedList, neighbor) and (not neighbor in closedList):
            coord = neighbor
            
            # print('search[coord] is ', search[coord])
            if search[coord] < counter:
                gValue[coord] = float('inf')
                search[coord] = counter
            if gValue[coord] > curCell.gValue + 1:
                gValue[coord] = curCell.gValue + 1
                fVal = gValue[coord] + hValue(coord, goal)
                newCell = Cell(coord, fVal, gValue[coord])
                if newCell in openList:
                    openList.remove(newCell)
                    BH_s.sort(openList)
                path[newCell.coord] = curCell
                # print('new cell is ', newCell.coord)
                BH_s.insert(newCell, openList)

# break ties in favor of smaller g-values
def aStar_gs(start, goal, maze, blockedList):
    counter = 0
    search = np.zeros((num_rows, num_cols))
    startHValue = hValue(start, goal)
    startCell = Cell(start, startHValue, 0)
    gValue = np.zeros((num_rows, num_cols))
    while startCell.coord != goal:
        gValue[startCell.coord] = 0
        gValue[goal] = float('inf')
        counter = counter + 1
        search[startCell.coord] = counter
        search[goal] = counter
        # print('search 22 is ',search[2][2])
        openList = []
        closedList = []
        # print('closed list ',closedList )
        BH_s.insert(startCell, openList)
        path = findRoute_gs(goal, openList, closedList, search, counter, blockedList, gValue)
        # expandedCells = expandedCells + len(closedList)
        # print(expandedCells)
        if not path:
            print("There is no path from startpoint to goal")
            return False
        pathList = []
        currCoord = goal
        while currCoord != startCell.coord:
            # print(currCoord, '->', end='')
            pathList.append(currCoord)
            currCoord = path[currCoord].coord
        # pathList.append(start)
        # print(startCell.coord, end='')
        pathList.reverse()
        for coord in pathList:
            if maze[coord] == 0:
                blockedList.append(coord)
                startCell = path[coord]
                # maze[coord] = 3
                # print('start cell is ', startCell.coord)
                break
            else:
                maze[coord] = 3
                # print('->',coord, end='')
                startCell.coord = goal
        # print(goal, '->', end='')
        # curCell = path[goal]
        # while curCell.coord != start:
        #     print(curCell.coord, '->', end='')
        #     curCell = path[curCell.coord]
        # print(start)
        # startCell.coord = goal
        # print('start coord now is ', startCell.coord)
 
def findRoute_gl(goal, openList, closedList, search, counter, blockedList, gValue):
    # print('search is ',search)
    # print('-------------------')
    curCell = BH_l.pop(openList)
    path = {}
    if(not curCell):
        return False
    # print(curCell.coord)
    closedList.append(curCell.coord)
    while curCell.coord != goal:
        # closedList.append(curCell)
        findNeighbors_gl(curCell, openList, goal, search, counter, path, blockedList, closedList, gValue)
        curCell = BH_l.pop(openList)
        if(not curCell):
            return False
        closedList.append(curCell.coord)
        # print(curCell.coord)
    # print('end find route -----------------')
    return path
    
def findNeighbors_gl(curCell, openList, goal, search, counter, path, blockedList, closedList, gValue):
    x = curCell.coord[0]
    y = curCell.coord[1]
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    for neighbor in neighbors:
        if isValid(neighbor[0], neighbor[1]) and not isBlocked(blockedList, neighbor) and (not neighbor in closedList):
            coord = neighbor
            
            # print('search[coord] is ', search[coord])
            if search[coord] < counter:
                gValue[coord] = float('inf')
                search[coord] = counter
            if gValue[coord] > curCell.gValue + 1:
                gValue[coord] = curCell.gValue + 1
                fVal = gValue[coord] + hValue(coord, goal)
                newCell = Cell(coord, fVal, gValue[coord])
                if newCell in openList:
                    openList.remove(newCell)
                    BH_l.sort(openList)
                path[newCell.coord] = curCell
                # print('new cell is ', newCell.coord)
                BH_l.insert(newCell, openList)

# break ties in favor of larger g-values
def aStar_gl(start, goal, maze, blockedList):
    counter = 0
    search = np.zeros((num_rows, num_cols))
    startHValue = hValue(start, goal)
    startCell = Cell(start, startHValue, 0)
    gValue = np.zeros((num_rows, num_cols))
    while startCell.coord != goal:
        gValue[startCell.coord] = 0
        gValue[goal] = float('inf')
        counter = counter + 1
        search[startCell.coord] = counter
        search[goal] = counter
        # print('search 22 is ',search[2][2])
        openList = []
        closedList = []
        # print('closed list ',closedList )
        BH_l.insert(startCell, openList)
        path = findRoute_gl(goal, openList, closedList, search, counter, blockedList, gValue)
        # expandedCells = expandedCells + len(closedList)
        # print(expandedCells)
        if not path:
            print("There is no path from startpoint to goal")
            return False
        pathList = []
        currCoord = goal
        while currCoord != startCell.coord:
            # print(currCoord, '->', end='')
            pathList.append(currCoord)
            currCoord = path[currCoord].coord
        # pathList.append(start)
        # print(startCell.coord, end='')
        pathList.reverse()
        for coord in pathList:
            if maze[coord] == 0:
                blockedList.append(coord)
                startCell = path[coord]
                # maze[coord] = 3
                # print('start cell is ', startCell.coord)
                break
            else:
                maze[coord] = 3
                # print('->',coord, end='')
                startCell.coord = goal
        # print(goal, '->', end='')
        # curCell = path[goal]
        # while curCell.coord != start:
        #     print(curCell.coord, '->', end='')
        #     curCell = path[curCell.coord]
        # print(start)
        # startCell.coord = goal
        # print('start coord now is ', startCell.coord)

def main():
    # initialize 
    maze = MG.generateMaze(num_rows, num_cols)
    start = (random.randint(0, num_rows-1), random.randint(0, num_cols-1))
    print("startpoint is ", start)
    goal = (random.randint(0, num_rows-1), random.randint(0, num_cols-1))
    print("goalpoint is ", goal)

    # maze = np.ones((num_rows, num_cols))
    # maze[(1,2)] = 0
    # maze[(2,2)] = 0
    # maze[(2,3)] = 0
    # maze[(3,2)] = 0
    # maze[(3,3)] = 0
    # maze[(4,3)] = 0
    # start = (4, 2)
    # goal = (4, 4)
    if maze[start[0]][start[1]] == 0 or maze[goal[0]][goal[1]] == 0:
        print("there is no path from startpoint to goal")
        return
    elif start == goal:
        print("start point is the same as goal")
        return
    blockedList = []
    # expandedCells_For = 0
    # expandedCells_back = 0
    startTime1 = time.time()
    aStar_gs(start, goal, maze, blockedList)
    endTime1 = time.time()

    startTime2 = time.time()
    aStar_gl(start, goal, maze, blockedList)
    endTime2 = time.time()

    maze[start] = 2
    maze[goal] = 4

    for l in range(0, num_rows+2):
        print('w', end='')
    print('')
    for i in range(0, num_rows):
        print('w', end='')
        for j in range(0, num_cols):
            if(maze[(i,j)])==0:
                print('x', end='')
            elif (maze[(i,j)]) == 1:
                print(' ', end='')
            elif (maze[(i, j)]) == 2:
                print('s', end='')
            elif (maze[(i, j)]) == 3:
                print('p', end='')
            else:
                print('t', end='')
        print('w')
    for l in range(0, num_rows+2):
        print('w', end='')
    print('')

    runtime1 = endTime1 - startTime1
    runtime2 = endTime2 - startTime2
    print("runtime in favor of smaller g-values: {}, runtime in favor of larger g-values: {}".format(runtime1, runtime2))

main()