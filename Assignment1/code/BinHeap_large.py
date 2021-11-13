# class Cell:
#     def __init__(self, coord, fValue, gValue):
#         self.coord = coord
#         self.fValue = fValue
#         self.gValue = gValue

# insert function for breaking ties with larger g-value
def insert(cell, openList):
    openList.append(cell)
    size = len(openList)
    i = size-1
    while i > 0:
        if openList[i].fValue < openList[(i-1) // 2].fValue:
            tmp = openList[(i-1) // 2]
            openList[(i-1) // 2] = openList[i]
            openList[i] = tmp
        # two cells have the same f-value
        if openList[i].fValue == openList[(i-1) // 2].fValue:
            # break ties with bigger g-value
            if openList[i].gValue > openList[(i-1) // 2].gValue:
                tmp = openList[(i-1) // 2]
                openList[(i-1) // 2] = openList[i]
                openList[i] = tmp
        i = i // 2

def sort(openList):
    size = len(openList)
    i = 0
    while (i * 2) < size:
        left = i*2 + 1
        right = i*2 + 2
        if left >= size:
            smallerChild = i*2
        else:
            if right < size:
                if openList[left].fValue < openList[right].fValue:
                    smallerChild = left
                elif openList[left].fValue == openList[right].fValue:
                    if openList[left].gValue >= openList[right].gValue:
                        smallerChild = left
                    else:
                        smallerChild = right
                else:
                    smallerChild = right
            else:
                smallerChild = left
        if openList[i].fValue > openList[smallerChild].fValue:
            tmp = openList[i]
            openList[i] = openList[smallerChild]
            openList[smallerChild] = tmp
        # two cells have the same f-value
        if openList[i].fValue == openList[smallerChild].fValue:
            # break ties with bigger g-value
            if openList[i].gValue < openList[smallerChild].gValue:
                tmp = openList[i]
                openList[i] = openList[smallerChild]
                openList[smallerChild] = tmp
        i = smallerChild

# pop function to pop the min cell
def pop(openList):
    size = len(openList)
    if size == 0:
        return False
    minCell = openList[0]
    openList[0] = openList[-1]
    openList.pop()
    if size > 2:
        sort(openList)
    return minCell

# cell1 = Cell((2,1), 2, 0)
# cell2 = Cell((2,5), 1, 0)
# cell3 = Cell((4,5), 1, 5)
# cell4 = Cell((4,7), 5, 2)
# cell6 = Cell((3,7), 3, 2)
# cell5 = Cell((3,4), 3, 3)

# openList = []
# insert(cell1, openList)
# insert(cell2, openList)
# insert(cell3, openList)
# insert(cell4, openList)
# insert(cell5, openList)
# insert(cell6, openList)

# for cell in openList:
#     print(cell.coord)
# print("------------------")
# for i in range(0, len(openList)):
#     print(pop(openList).coord)