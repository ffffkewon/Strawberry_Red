import sys
a=[]
T = int(input())
for i in range(0,T):
    a.append(str(sys.stdin.readline()))
next

score_list = []
score = 0
for i in range(0,T):
    b = list(a[i])
    for j in range(0,len(b)):
        if b[j] == 'O':
            score += 1
            score_list.append(score)
        elif b[j] == 'X':
            score = 0
            score_list.append(score)
    next
    score_sum = sum(score_list)
    print(score_sum)
    score_list = []
next


#for i in range(0,T):
#    b = list(a[i])
#    score = b.count('O')
#    if b[-2] == 'X':
#        score = score
#    elif b[-2] == 'O':
#        score += 2
#    print(score)
#next

