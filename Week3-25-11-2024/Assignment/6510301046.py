def find_coeff(x: list, y: list):

    if len(x) != len(y):

        raise Exception("X and Y have different size.")

    sxx = 0
    sxy = 0

    # Find mean of x and y
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)

    # Find Sxx and Sxy for coefficient a
    for i, j in zip(x, y):
        sxx += (i-mean_x)**2
        sxy += (i-mean_x)*(j-mean_y)

    # Calculate coefficient a
    a = round(sxy / sxx, ndigits=2)

    # Calculate coefficient b
    b = round(mean_y - mean_x*a, ndigits=2)

    return a, b


    
if __name__ == "__main__":
    x = [29, 28, 34, 31, 25]
    y = [77, 62, 93, 84, 59]


    a, b = find_coeff(x, y)

    print(a, b)