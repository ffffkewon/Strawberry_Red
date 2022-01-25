import sys

N, X = map(int, sys.stdin.readline().split())
a = list([int(x) for x in input().split()])

b=[]

for j in range(0,len(a)):
    if a[j] < X:
        print(a[j],end=" ")
    
next
print("")

