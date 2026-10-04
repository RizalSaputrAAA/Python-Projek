mat1 = [
    [1, 5],
    [9, 10],
]

mat2 = [
    [9, 15],
    [19, 101],
]

for x in range(0, len(mat1)):
    for y in range(0, len(mat1[0])):
        print(mat1[x][y] + mat2[x][y], end=' ')
    print()
