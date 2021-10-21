# insert function for breaking ties with smaller g-value
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
                    if openList[left].gValue <= openList[right].gValue:
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