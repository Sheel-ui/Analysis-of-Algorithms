import sys

# Task 1 - Brute force for Problem 1 in O(m^3n^3)
def alg1(matrix,h):
    i1,j1,i2,j2 = None, None, None, None
    maxSquareSize = 0
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        for j in range(cols):
            # above two for loops for selecting element in matrix
            for k in range(rows):
                for l in range(cols):
                    # above two for loops for selecting corners of submatrix
                    squareSize = 0
                    if k-i == l-j and k-i>=0:
                        squareSize = k-i+1
                        flag = True
                        for x in range(i, i+squareSize):
                            for y in range(j, j+squareSize):
                                # above two for loops for checking each element in submatrix
                                if matrix[x][y] < h:
                                    flag = False
                        if flag:
                            if squareSize >= maxSquareSize:
                                maxSquareSize = squareSize
                                i1, j1 = i+1, j+1
                                i2, j2 = i+maxSquareSize, j+maxSquareSize
    return i1,j1,i2,j2


# Task 2 - Brute force for Problem 1 in O(m^2n^2)
def alg2(matrix, h):
    rows, cols = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0
    maxSquareSize = 0
    i1, j1, i2, j2 = None, None, None, None
    for i in range(rows):
        for j in range(cols):
            # above two for loops for selecting element in matrix
            if matrix[i][j] >= h:
                squareSize = 1
                flag = True
                while squareSize + i < rows and squareSize + j < cols and flag:
                    # above while loop for incrementing diagonal element
                    for k in range(j, squareSize + j + 1):
                        # above for loop for checking every element in a column including diagonal element
                        if matrix[i + squareSize][k] < h:
                            flag = False
                    for k in range(i, squareSize + i + 1):
                        # above for loop for checking every element in a row including diagonal element
                        if matrix[k][j + squareSize] < h:
                            flag = False
                    if flag:
                        squareSize += 1
                if squareSize >= maxSquareSize:
                    maxSquareSize = squareSize
                    i1, j1 = i+1, j+1
                    i2, j2 = i+maxSquareSize, j+maxSquareSize
    return i1, j1, i2, j2


# Task 3 - DP for Problem 1 in O(mn)
def alg3(matrix, h):
    rows, cols = len(matrix), len(matrix[0])

    # Initialize dp array with 0
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]

    # To store maximum square size
    maxSquareSize = 0

    # To store bounding indices of the max square
    i1,j1,i2,j2 = None, None, None, None

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if matrix[i - 1][j - 1] >= h:
                # bellman equation: value of current element = minimum value of top, left, diagonal element + 1
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
                if dp[i][j] >= maxSquareSize:
                    # keeping track of max size in same loop
                    maxSquareSize = dp[i][j]
                    i1,j1 = i-maxSquareSize + 1, j - maxSquareSize + 1
                    i2,j2 = i,j
    return  i1, j1, i2, j2


# Task 4 - DP for Problem 2 in O(mn^2)
def alg4(matrix, h):
    m, n = len(matrix), len(matrix[0])

    # converting elements under threshold to 0 and rest as 1
    binaryMatrix = [[0] * (n) for _ in range(m)]
    for i in range(m):
        for j in range(n):
            binaryMatrix[i][j] = 0 if matrix[i][j]<h else 1
    
    # create prefix sum
    prefixSum = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefixSum[i][j] = prefixSum[i - 1][j] + prefixSum[i][j - 1] - prefixSum[i - 1][j - 1] + binaryMatrix[i - 1][j - 1]

    i1,j1,i2,j2 = 0,0,0,0
    maxSquareSize = 0
    for i in range(m):
        for j in range(n):
            # above two for loops for selecting an element
            for k in range(1,min(m-i,n-j)+1):
                # above for loop for incrementing the diagonal element
                subSquareSum = prefixSum[i+k][j+k] - prefixSum[i][j+k] - prefixSum[i+k][j] + prefixSum[i][j]
                cornerSum = 0
                if k>1:
                    cornerSum = binaryMatrix[i][j] + binaryMatrix[i+k-1][j+k-1] + binaryMatrix[i][j+k-1] + binaryMatrix[i+k-1][j]
                else:
                    cornerSum = binaryMatrix[i][j]
                # inside sum is sum of subsquare minus the sum of corners, that should be greater than max no of 1s -4 1s belonging to corner
                insideSum = subSquareSum-cornerSum
                if insideSum>=(k*k-4):
                    if k>=maxSquareSize:
                        maxSquareSize = k
                        i1, j1 = i+1, j+1
                        i2, j2 = i1+k-1, j1+k-1
    return i1,j1,i2,j2


# Task 5A - Top Down Recursion + Memoization for Problem 2 in O(mn)
def alg5a(matrix, h):
    def calculateTopLeft(matrix, top_left, row, col, h):
        # Check if original element is 0
        if matrix[row][col] < h:
            top_left[row][col] = 1
        # check if top-right or bottom-left are 0 in original matrix
        elif matrix[row][col-1] < h or matrix[row-1][col] < h:
            top_left[row][col] = 1
        else:
        # make recursive call
            if top_left[row][col-1] == -1:
                    calculateTopLeft(matrix, top_left, row, col-1, h)
            if top_left[row-1][col] == -1:
                    calculateTopLeft(matrix, top_left, row-1, col, h)
            if top_left[row-1][col-1] == -1:
                    calculateTopLeft(matrix, top_left, row-1, col-1, h)
            top_left[row][col] = min(top_left[row][col-1], top_left[row-1][col], top_left[row-1][col-1])+1

    def calculateTopRight(matrix, top_right, row, col, h):
            # Check if original element is 0
            if matrix[row][col] < h:
                top_right[row][col] = 1
            # check if top-left or bottom-left are 0 in original matrix
            elif matrix[row-1][col-1] < h or matrix[row][col-1] < h:
                top_right[row][col] = 1
            else:
            # make recursive call
                if top_right[row][col-1] == -1:
                        calculateTopRight(matrix, top_right, row, col-1, h)
                if top_right[row-1][col] == -1:
                        calculateTopRight(matrix, top_right, row-1, col, h)
                if top_right[row-1][col-1] == -1:
                        calculateTopRight(matrix, top_right, row-1, col-1, h)
                top_right[row][col] = min(top_right[row][col-1], top_right[row-1][col], top_right[row-1][col-1])+1

    def calculateBottomLeft(matrix, bottom_left, row, col, h):
            # Check if original element is 0
            if matrix[row][col] < h:
                bottom_left[row][col] = 1
            # check if top-left or top_right are 0 in original matrix
            elif matrix[row-1][col-1] < h or matrix[row-1][col] < h:
                bottom_left[row][col] = 1
            else:
            # make recursive call
                if bottom_left[row][col-1] == -1:
                        calculateBottomLeft(matrix, bottom_left, row, col-1, h)
                if bottom_left[row-1][col] == -1:
                        calculateBottomLeft(matrix, bottom_left, row-1, col, h)
                if bottom_left[row-1][col-1] == -1:
                        calculateBottomLeft(matrix, bottom_left, row-1, col-1, h)
                bottom_left[row][col] = min(bottom_left[row][col-1], bottom_left[row-1][col], bottom_left[row-1][col-1])+1
    i1, j2, i2, j2 = 0,0,0,0
    rows, cols = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0
    dp = [[0]*cols for _ in range(rows)]
    top_right = [[-1]*cols for _ in range(rows)]
    top_left = [[-1]*cols for _ in range(rows)]
    bottom_left = [[-1]*cols for _ in range(rows)]

    # Iterate each element
    for i in range(rows):
        for j in range(cols):
            # initializing corner values
            if i==0 or j==0:
                dp[i][j] = 1
                top_left[i][j] = 1
                top_right[i][j] = 1
                bottom_left[i][j] = 1
            else:
                # else recursively calling DP values
                if top_left[i-1][j-1] == -1:
                    calculateTopLeft(matrix, top_left, i-1, j-1, h)
                if top_right[i-1][j] == -1:
                    calculateTopRight(matrix, top_right, i-1, j, h)
                if bottom_left[i][j-1] == -1:
                    calculateBottomLeft(matrix, bottom_left, i, j-1, h)
                dp[i][j] = min(top_left[i-1][j-1], top_right[i-1][j], bottom_left[i][j-1])+1
    
    maxSquareSize = 0
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            # single pass in resultant matrix to get the solution
            if dp[i][j] >= maxSquareSize:
                maxSquareSize = dp[i][j]
                i2,j2 = i+1,j+1
                i1,j1 = i2-maxSquareSize+1, j2-maxSquareSize+1
    return i1,j1,i2,j2


# Task 5B - Bottom Up DP for Problem 2 in O(mn)
def alg5b(matrix, h):
    rows, cols = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0

    # Pad original matrix with 0's for simplicity
    for i in range(rows):
        matrix[i].insert(0,0)
    matrix.insert(0,[0 for i in range(cols+1)])

    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    top = [[0] * (cols + 1) for _ in range(rows + 1)]
    left = [[0] * (cols + 1) for _ in range(rows + 1)]
    top_left = [[0] * (cols + 1) for _ in range(rows + 1)]

    # Calculating dp ignoring top-right
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            if i == 1 or j == 1:
                top[i][j] = 1
            else:
                # Check if original element is <h
                if matrix[i][j] < h:
                    top[i][j] = 1
                # check if top-left or bottom-left are <h in original matrix
                elif matrix[i-1][j-1] < h or matrix[i][j-1] < h:
                    top[i][j] = 1
                else:
                    top[i][j] = min(top[i][j-1], top[i-1][j], top[i-1][j-1])+1
    
    # Calculating dp ignoring bottom-left
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            if i == 1 or j == 1:
                left[i][j] = 1
            else:
                # Check if original element is <h
                if matrix[i][j] < h:
                    left[i][j] = 1
                # check if top-left or top-right are <h in original matrix
                elif matrix[i-1][j-1] < h or matrix[i-1][j] < h:
                    left[i][j] = 1
                else:
                    left[i][j] = min(left[i][j-1], left[i-1][j], left[i-1][j-1])+1

    # Calculating dp ignoring top-left
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            if i == 1 or j == 1:
                top_left[i][j] = 1
            else:
                # Check if original element is <h
                if matrix[i][j] < h:
                    top_left[i][j] = 1
                # check if top-right or bottom-left are <h in original matrix
                elif matrix[i][j-1] < h or matrix[i-1][j] < h:
                    top_left[i][j] = 1
                else:
                    top_left[i][j] = min(top_left[i][j-1], top_left[i-1][j], top_left[i-1][j-1])+1
    
    maxSquareSize = 0
    i1, j1, i2, j2 = None, None, None, None
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            # single pass in resultant matrix to get the solution
            if i == 1 or j == 1:
                dp[i][j] = 1
            else:
                dp[i][j] = min(top_left[i-1][j-1], top[i-1][j], left[i][j-1])+1
            if dp[i][j] >= maxSquareSize:
                maxSquareSize = dp[i][j]
                i1,j1 = i-maxSquareSize + 1, j - maxSquareSize + 1
                i2,j2 = i,j
    return i1,j1,i2,j2


# Task 6 - Brute force for Problem 3 in O(m^3n^3)
def alg6(matrix,h,k):
    max_allowed_less_than_threshold = k
    i1,j1,i2,j2 = None, None, None, None
    maxSquareSize = 0
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            # above for loop for selecting each element
            for k in range(rows):
                for l in range(cols):
                    # above for loops for selecting each corner of submatrix
                    squareSize = 0
                    # calculating number of elements less than threshold
                    if k-i == l-j and k-i>=0:
                        squareSize = k-i+1
                        less_than_threshold_count = 0
                        for x in range(i, i+squareSize):
                            for y in range(j, j+squareSize):
                                if matrix[x][y] < h:
                                    less_than_threshold_count += 1
                        # checking if number of elements less than threshold is a less max allowed elements
                        if less_than_threshold_count<=max_allowed_less_than_threshold:
                            if squareSize >= maxSquareSize:
                                maxSquareSize = squareSize
                                i1, j1 = i+1, j+1
                                i2, j2 = i+maxSquareSize, j+maxSquareSize
    return i1,j1,i2,j2


def transformList(i,j,k_list):
    # transforming the perspective of 0s according to index of current element
    return [(k[0]-i,k[1]-j) for k in k_list if (k[0] -i >=0 and k[1]-j>=0)]


def prefixSum(i,j,matrix,prefixSumDP):
    # recursive code to generate prefix sum matrix
    if i == 0 or j == 0:
        prefixSumDP[i][j]=0
        return 0
    elif prefixSumDP[i][j] != -1:
        return prefixSumDP[i][j]
    else:
        prefixSumDP[i][j] = prefixSum(i - 1,j,matrix,prefixSumDP) + prefixSum(i,j-1,matrix,prefixSumDP) - prefixSum(i-1,j-1,matrix,prefixSumDP) + matrix[i - 1][j - 1]
        return prefixSumDP[i][j]


# Task 7A - Top Down Recursion + Memoization for Problem 3 in O(mnk)
def alg7a(matrix,h,k):
    matrix = [[0 if i<h else 1 for i in j] for j in matrix]
    m = len(matrix)
    n = len(matrix[0])
    k_list = []
    dp = [[0]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            if matrix[i][j]==0:
                k_list.append((i,j))

    prefixSumDP = [[-1]*(n+1) for i in range(m+1)]
    for i in range(m):
        for j in range(n):
            # above two for loops to select an element in matrix
            flag = False
            max_size = 0
            max_zeros = m*n+1
            for z in transformList(i,j,k_list):  
                # above for loop for selecting k zeros that are allowd
                size = max(z[0],z[1]) + 1
                if i+size <= m and j+size <= n:
                    # for every element checking the subsquare sum is 0s less than allowed k
                    # making RECURSIVE calls to generate prefix sum DP
                    subSquareSum = prefixSum(i+size,j+size,matrix,prefixSumDP) - prefixSum(i,j+size,matrix,prefixSumDP)  - prefixSum(i+size,j,matrix,prefixSumDP) + prefixSum(i,j,matrix,prefixSumDP)
                    zeros = size*size-subSquareSum
                    if zeros>k and zeros<max_zeros:
                        flag=True
                        max_size = size
                        max_zeros = zeros
            if not flag:
                dp[i][j]=min(m-i,n-j)
            else:
                dp[i][j]=max_size-1
    x,y,size=0,0,0
    for i in range(m):
        for j in range(n):
            # single pass to get the max solution
            if dp[i][j]>=size:
                x,y=i+1,j+1
                size = dp[i][j]
    if x+size-1 >= x or y+size-1 >= y:
        return x,y, x+size-1, y+size-1
    else:
        return None, None, None, None


# Task 7B - Bottom Up DP for Problem 3 in O(mnk)
def alg7b(matrix,h,k):
    matrix = [[0 if i<h else 1 for i in j] for j in matrix]
    m = len(matrix)
    n = len(matrix[0])
    k_list = []
    dp = [[0]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            if matrix[i][j]==0:
                k_list.append((i,j))
    prefixSum = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # populating the values of prefixsum FP
            prefixSum[i][j] = prefixSum[i - 1][j] + prefixSum[i][j - 1] - prefixSum[i - 1][j - 1] + matrix[i - 1][j - 1]
    for i in range(m):
        for j in range(n):
            # above two for loops to select an element in matrix
            flag = False
            max_size = 0
            max_zeros = m*n+1
            for z in transformList(i,j,k_list): 
                # above for loop for selecting k zeros that are allowd 
                size = max(z[0],z[1]) + 1
                if i+size <= m and j+size <= n:
                    # for every element checking the subsquare sum is 0s less than allowed k
                    # using precomputed DP matrix to get the sub square sum
                    subSquareSum = prefixSum[i+size][j+size] - prefixSum[i][j+size] - prefixSum[i+size][j] + prefixSum[i][j]
                    zeros = size*size-subSquareSum
                    if zeros>k and zeros<max_zeros:
                        flag=True
                        max_size = size
                        max_zeros = zeros
            if not flag:
                dp[i][j]=min(m-i,n-j)
            else:
                dp[i][j]=max_size-1
    x,y,size=0,0,0
    for i in range(m):
        for j in range(n):
            # single pass to get the max solution
            if dp[i][j]>=size:
                x,y=i+1,j+1
                size = dp[i][j]

    if x+size-1 >= x or y+size-1 >= y:
        return x,y, x+size-1, y+size-1
    else:
        return None, None, None, None


if __name__ == "__main__":

    m_n_h_k = input()
    args = m_n_h_k.split(' ')
    m,n,h,k = int(args[0]), int(args[1]), int(args[2]), 0
    if len(args)>3:
        k = int(args[3])

    matrix = []
    for i in range(m):
        row = input()
        matrix.append([int(value) for value in row.split(' ')])


    # Invoking Strategies

    if sys.argv[1] == "alg1":
        output = alg1(matrix,h)
    elif sys.argv[1] == "alg2":
        output = alg2(matrix, h)
    elif sys.argv[1] == "alg3":
        output = alg3(matrix, h)
    elif sys.argv[1] == "alg4":
        output = alg4(matrix, h)
    elif sys.argv[1] == "alg5a":
        output = alg5a(matrix, h)
    elif sys.argv[1] == "alg5b":
        output = alg5b(matrix, h)
    elif sys.argv[1] == "alg6":
        output = alg6(matrix,h,k)
    elif sys.argv[1] == "alg7a":
        output = alg7a(matrix,h,k)
    elif sys.argv[1] == "alg7b":
        output = alg7b(matrix,h,k)


    print(" ".join(str(idx) for idx in output))