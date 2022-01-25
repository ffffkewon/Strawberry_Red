N = int(input())

n = 111

a, b = divmod(N,10)
i = 0
while N != n:
    if i == 0:
        a, b = divmod(N,10)
        n = N
    else:
        a, b = divmod(n,10)
    if n < 10:
        n = 10*b + b
    else:
        n = 10*b + (a+b) % 10
    i += 1
print(i)

