import sys

a = []
for i in range(0,9):
    a.append(int(sys.stdin.readline()))
next

max_value = max(a)
max_index = a.index(max_value)+1

print(max_value)
print(max_index)
