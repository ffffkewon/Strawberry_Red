import sys
import random

N, X = map(int, sys.stdin.readline().split())
a = []
for i in range(0,N):
    a.append(random.randint(1,10000))
next
b=[]

for j in range(0,len(a)):
    if a[j] <= X:
        print(a[j],end=" ")
next

