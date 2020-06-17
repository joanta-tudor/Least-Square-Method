import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read1_csv(path):
    df = pd.read_csv(path)
    process(df)

def process(df):
    x = df.drop(
        ["Country", "Happiness.Rank", "Whisker.high", "Whisker.low", "Family", "Health..Life.Expectancy.", "Generosity",
         "Trust..Government.Corruption.", "Dystopia.Residual"], axis=1)
    y = df['Happiness.Score']

    x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.85, random_state=42, shuffle=False)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    for i in range (0,len(x_train)):
        x_train[i][0]=1
    linear_regr(x_train,y_train)

def linear_regr(x_train,y_train):
    xx = []
    for i in x_train:
        i.insert(0,1)
    b=inmultire(Inversa(inmultire(transpus(x_train),x_train)),transpus(x_train))
    ynou=[]
    for i in range(0,len(y_train)):
        ynou.append([y_train.array[i]])
    rez=inmultire(b,ynou)
    for x in rez:
        xx.append(x[0])
    return xx

#transpusa unei matrici
def transpus(m):
    rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return  rez

#calculam inversa unei matrici
def Minor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def Deternminant(m):
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinan = 0
    for c in range(len(m)):
        determinan += ((-1)**c)*m[0][c]*Deternminant(Minor(m,0,c))
    return determinan

def Inversa(m):
    determinant = Deternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = Minor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * Deternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transpus(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

#inmultire a doua matrici
def inmultire(m1,m2):

    rez=[[0 for x in range(len(m2[0]))] for y in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range (len(m2)):
                rez[i][j]+=m1[i][k]*m2[k][j]
    return rez

