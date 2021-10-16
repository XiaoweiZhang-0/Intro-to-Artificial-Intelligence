# binary min heap
import random

# insert function for breaking ties with smaller g-value
def insert(cell, openList):
    i = len(openList)
    while i // 2 > 0:
        if openList[i].fValue < openList[i // 2].fValue:
            tmp = openList[i // 2]
            openList[i // 2] = openList[i]
            openList[i] = tmp
        # two cells have the same f-value
        if openList[i].fValue == openList[i // 2].fValue:
            # break ties with smaller g-value
            if openList[i].gValue < openList[i // 2].gValue:
            # break ties with bigger g-value
            # if openList[i].gValue < openList[i // 2].gValue:
                tmp = openList[i // 2]
                openList[i // 2] = openList[i]
                openList[i] = tmp
        i = i // 2
    openList[i] = cell

def sort(openList, i):
    size = len(openList)
    while (i * 2) <= size:
        if i*2+1 > size:
            sid = i*2 # smaller child
        else:
            if openList[i*2].fValue < openList[i*2+1].fValue:
                sid = i*2
            else:
                sid = i*2+1
        if openList[i].fValue > openList[sid].fValue:
            tmp = openList[i]
            openList[i] = openList[sid]
            openList[sid] = tmp
        # two cells have the same f-value
        if openList[i].fValue == openList[sid].fValue:
            # break ties with smaller g-value
            if openList[i].gValue > openList[sid].gValue:
            # break ties with bigger g-value
            # if openList[i].gValue < openList[i // 2].gValue:
                tmp = openList[i]
                openList[i] = openList[sid]
                openList[sid] = tmp
        i = sid

# pop function to pop the min cell
def pop(openList):
    size = len(openList)
    if size == 0:
        return False
    minCell = openList[0]
    openList[0] = openList[-1]
    openList.pop()
    sort(openList, 0)
    return minCell