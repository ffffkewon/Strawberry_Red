import sys
numbers = []
for i in range(10):
    numbers.append(int(sys.stdin.readline()))
next

remains = [numbers[j]%42 for j in range(len(numbers))]
remains = list(set(remains))


print(len(remains))


