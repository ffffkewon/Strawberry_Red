

N = int(input())

a = [1,2,3,4,5,6,7,8,9]
b = [N*a[j] for j in range(0,len(a))]
for i in range(0,len(a)):

    print(N,"*",a[i],"=",b[i])
next
