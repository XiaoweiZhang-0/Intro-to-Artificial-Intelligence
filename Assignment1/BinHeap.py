# binary min heap
import random

# insert function
def insert(f_value, list, dict, cell):
    if f_value not in list:
        dict[f_value] = [cell]
        i = len(list)
        while i // 2 > 0:
            if f_value < list[i // 2]:
                temp = list[i // 2]
                list[i // 2] = list[i]
                list[i] = temp
            i = i // 2
        list[i] = f_value
    else:
        # f-value already exists in the list
        dict[f_value].append(cell)

# sort function to minify the heap
def sort(list, i):
    sorted = i
    left = i*2 + 1 # left child
    right = i*2 + 2 # right child
    if left < len(list) -1 and list[left] < list[i]:
        sorted = left
    if right < len(list) -1 and list[right] < list[sorted]:
        sorted = right
    if sorted != i:
        list[sorted], list[i] = list[i], list[sorted]
        sort(list,sorted)

# pop function
def pop(list, dict):
    cell = dict[list[0]].pop(random.randrange(len(dict[list[0]])))
    if len(dict[list[0]]) == 0:
        del dict[list[0]]
        list[0] = list[-1]
        list.pop()
        sort(list,0)
    return cell
