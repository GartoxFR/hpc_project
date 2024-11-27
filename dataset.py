from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0

train_X_flat = train_X.flatten().reshape(60000,784)
test_X_flat = test_X.flatten().reshape(10000,784)

with open("mnist.txt", "w") as file:
    for i in range(len(train_X_flat)):
        file.write("[")
        for j in range(len(train_X_flat[i])):
            file.write(str(train_X_flat[i][j]))
            if(j+1 != len(train_X_flat[j])):
                file.write(",")
        file.write("],")
        file.write(str(train_y[i]))
        if i + 1 != len(train_X_flat):
            file.write("\n")