import numpy as np


if __name__ == "__main__":
    framenames = [1, 2, 3, 4, 5, 6, 7]
    for i in range(0, len(framenames), 6):
        chunk = framenames[i:i + 6]
        print(chunk)
