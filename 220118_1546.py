N = int(input())

exam = list([int(x) for x in input().split()])

M = max(exam)


for i in range(0,len(exam)):
    exam[i] = exam[i]/M * 100
next

new_avg = sum(exam) / len(exam)

print(new_avg)

