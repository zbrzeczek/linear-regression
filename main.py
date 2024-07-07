import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_path = "train.csv"
test_path = "test.csv"

train_read = pd.read_csv(train_path)
test_read = pd.read_csv(test_path)


# z pandas moizesz zrobic .nazwa_kolumny i bedzie ci zczytywac te dane wiec nie trzeba brac z csv tej kolumny specjalnie
#year_column = "model_year" 
#target_column = "price"

#narysowanie danych
#plt.scatter(train_read.model_year, train_read.price)
#plt.show()

def lossFunction(m, b, points):
    totalError = 0
    for i in range(len(points)):
        x = points.iloc[i].model_year
        y = points.iloc[i].price
        totalError += (y - (m*x - b)) ** 2

    total = totalError/len(points)

def gradientDesc(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].model_year
        y = points.iloc[i].price
        m_gradient += x * (y - (m_now * x + b_now))
        b_gradient += y - (m_now * x + b_now)

    m_total = -2/n * m_gradient
    b_total = -2/n * b_gradient

    m = m_now - m_total * L
    b = b_now - b_total * L

    return m, b

m = 0
b = 0
L = 0.0001
epochs = 400

for i in range(epochs):
    if i%50==0:
        print("Epochs: {i}")
    m, b = gradientDesc(m, b, train_read, L)

plt.scatter(train_read.model_year, train_read.price, color="black")
plt.plot(list(range(1990, 2030)), [m*x + b for x in range(1990, 2030)], color="red")
plt.show()