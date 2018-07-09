import random

A = [1, 2, 3, 4]

def tre():
    yield random.choice(A)

print(next(tre()))
print(next(tre()))