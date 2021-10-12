

# calculate Manhattan distance as h value
def H_value(currCell, goalCell):
    return abs(currCell[0] - goalCell[0]) + abs(currCell[1] - goalCell[1])
