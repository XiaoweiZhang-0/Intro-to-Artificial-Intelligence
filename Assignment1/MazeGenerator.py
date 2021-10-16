#This is maze generator
import numpy as np
import random
import sys

num_rows = 101
num_cols = 101

np.set_printoptions(threshold=sys.maxsize)
def findNeighbor(row, col, unvisited, stack, maze):
    neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    # for (row, col) in neighbors:
    #     # print(row, col)
    #     if(row >= 0 and col >= 0 and (row, col) in unvisited and (row, col) not in stack):
    #         stack.append((row, col))
    while neighbors:
        row, col = random.choice(neighbors)
        rdm_num = random.randint(1,100)
        # print(rdm_num)        
        if(row >= 0 and col >= 0 and (row, col) in unvisited and (row, col) not in stack and rdm_num > 30):
            maze[(row, col)] = 1
            # print('appended')
            stack.append((row, col))
        neighbors.remove((row, col))
        if (row, col) in unvisited:
            unvisited.remove((row, col))
    
def generateMaze():
    maze = np.zeros((num_rows, num_cols))
    stack = []
    unvisited = list()
    for i in range(0 , num_rows):
        for j in range(0, num_cols):
            unvisited.append((i, j))
    while unvisited:
        # stack.append(unvisited.pop())
        startPoint = random.choice(unvisited)
        maze[startPoint] = 1
        unvisited.remove(startPoint)
        row = startPoint[0]
        col = startPoint[1]
        findNeighbor(row, col, unvisited, stack, maze)
        while stack:
            row, col = stack.pop()
            # unvisited.remove((row, col))
            # print('current cell is row:{}, col:{}' .format(row, col))
        # print(unvisited)
            findNeighbor(row, col, unvisited, stack, maze)
            # print('end search')
    # cnt_blk = 0
    # cnt_unblk = 0
    # for i in range(0, num_rows):
    #     for j in range(0, num_cols):
    #         if(maze[(i,j)])==0:
    #             cnt_blk = cnt_blk+1
    #             print('x', end='')
    #         else:
    #             cnt_unblk = cnt_unblk+1
    #             print(' ', end='')
    #     print('')
    # print('block ratio is ' + str(cnt_blk/(cnt_unblk+cnt_blk)))
    return maze


