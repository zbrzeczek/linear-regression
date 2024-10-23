import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# z pandas moizesz zrobic .nazwa_kolumny i bedzie ci zczytywac te dane wiec nie trzeba brac z csv tej kolumny specjalnie

class LinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None
        self.TSS = None # total sum of squares
        self.RSS = None # residual sum of squares
        self.Error = None # 1 - (rss/tss)
        self.residual = None

    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate the terms needed for the slope (b1) and intercept (bo) of the regression line 
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        # Calculate the slope (b1) and intercept (bo) of the regression line (regression equation)
        self.weight = numerator / denominator
        self.bias = y_mean - self.weight * x_mean
        
        y_pred = self.bias + self.weight * x
        self.residual = y - y_pred
        
        self.RSS = np.sum(self.residual ** 2)
        self.TSS = np.sum((y - y_mean) ** 2)
        self.r2score_ = 1 - (self.RSS / self.TSS)

    def output(self, x):
        x = np.array(x, dtype=float)
        return x * self.weight + self.bias


def main():
    train_path = "train.csv"
    test_path = "test.csv"

    train_read = pd.read_csv(train_path)
    #test_read = pd.read_csv(test_path)

    x = train_read.model_year
    y = train_read.price

    model = LinearRegression()
    model.fit(x, y)
    
    print("Wpisz rok auta")
    x_test = input()

    pred = model.output(x_test)

    print("Twoje auto bedzie kosztować około: ", pred)

    '''plt.figure(figsize = (8,6))
    plt.scatter(x, y, marker='o', color='red')
    plt.plot(test_read.model_year, pred, color='black',markerfacecolor='red',
             markersize=10,linestyle='solid')
    plt.xlabel("Roczniki")
    plt.ylabel("Ceny")
    plt.show()''' # rysowanie jak kto l;ubie

     
if __name__=="__main__":
    main()