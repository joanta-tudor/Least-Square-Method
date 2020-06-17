import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from service2 import linear_regr


def read_csv(path):
    df = pd.read_csv(path)
    procesare(df)


def procesare(df):
    x = df.drop(
        ["Country", "Happiness.Rank", "Whisker.high", "Whisker.low", "Family", "Health..Life.Expectancy.", "Generosity",
         "Trust..Government.Corruption.", "Dystopia.Residual"], axis=1)
    y = df['Happiness.Score']

    x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.85, random_state=42, shuffle=False)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    linear_mod(x_train, x_test, y_train, y_test, df)


def linear_mod(x_train, x_test, y_train, y_test, plot_y):
    lm = linear_model.LinearRegression()
    q = []
    for i in x_train:
        q.append([i[1],i[2]])
    lm.fit(q, y_train)
    print(lm.intercept_,lm.coef_)
    rez=linear_regr(q,y_train)
    #y_pred = lm.predict(list(x_train) + list(x_test))

    x=[]
    y=[]
    z=[]

    for i in range (0,len(x_train)):
        x.append(x_train[i][1])
        y.append(x_train[i][2])
        z.append(y_train.array[i])

    fig=plt.figure()
    xx=fig.add_subplot(111,projection='3d')
    xx.scatter(x,y,z,c='y',marker='o')

    #luam punctele de pe dreapta
    xref=[]
    yref=[]
    zref=[]
    valx=min(x)
    valy=min(y)
    stepx=(max(x)-valx)/len(x)
    stepy=(max(y)-valy)/len(y)
    matr=[]

    for i in range(len(x)):
        xref.append(valx)
        yref.append(valy)
        matr.append([valx,valy])
        valx+=stepx
        valy+=stepy


    for i in range(0,len(xref)):
        zref.append(lm.intercept_+xref[i]*lm.coef_[0]+yref[i]*lm.coef_[1])
    xx.plot(xref,yref,zref)
    plt.show()

    errtool=0
    err=0
    for i in range(0,len(x_test)):
        pr=rez[0]+x_test[i][1]*rez[1]+x_test[i][2]*rez[2]
        err+=(pr-y_test[i])**2
        errtool+=(lm.predict([[x_test[i][1],x_test[i][2]]])[0]-y_test[i])**2

    print("Eroare fara tool:" ,err/len(x_test))
    print("Eroare c tool:" ,errtool/len(x_test))

def plot_array(array):
    plt.plot(array)
    plt.show()
