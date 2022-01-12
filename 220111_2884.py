H, M = map(int, input().split())
M = M - 45
if M<0:
    if H == 0 :
        H = 23
        M = 60 + M
    else:
        H = H -1
        M = 60 + M

print(H,M)

