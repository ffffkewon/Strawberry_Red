import sys

N = int(input())
c = 0
for i in range(N):
    b = list([int(x) for x in input().split()])
    mean = sum(b) / len(b)
    for j in range(0,len(b)):
        if b[j] > mean:
            c += 1
    next
    portion = c / len(b) * 100

    print('{:.3f}%'.format(portion))

    c=0
    b = []

next
