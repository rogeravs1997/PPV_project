import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# LOADING THE DATA FROM A .MAT FILE (MATLAB/OCTAVE FILE)
file_name = "limpiado"
main_path = ("D:\Desktop\Modelo Predictivo PPV\database")
file_path = (file_name + ".xlsx")
sheet_name = "data"
dataset = pd.read_excel(main_path + "\\" + file_path, sheet_name)
dataset=dataset.sample(n=200,random_state=1)

# DEFINING FEATURES AND LABELS

X=dataset[["Distancia (m)","Kg de columna explosiva"]]
y=dataset["PPV"]

# Casting into numpy arrays
X=np.array(X,dtype=np.float64)
y=np.array(y,dtype=np.float64)


# SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3,shuffle=True)
y_train.shape, y_test.shape = (-1, 1), (-1, 1)

# SOLVING THE USBM FORMULA WE GET A MODEL OF FORM: log(y)=log(K)-(B*log(X1/sqrt(X2)))
# DEFINING A FUNCTION TO TRANSFORM THE DATA INTO DESIRE FORMAT
def log_transform(X,y):
    X_transform=np.log(X[:,0]/np.sqrt(X[:,1]))
    y_transform=np.log(y)
    X_transform.shape, y_transform.shape = (-1, 1), (-1, 1)
    
    return X_transform , y_transform
    
X_train_transform , y_train_transform = log_transform(X_train, y_train)
#add an extra column of 1's to X
X_train_transform=np.concatenate((np.ones((len(y_train_transform),1)),X_train_transform),axis=1)
model=LinearRegression(fit_intercept=False)
model.fit(X_train_transform,y_train_transform)


K=np.exp(model.coef_[0][0])
B=-1*model.coef_[0][1]


def predict_USBM ( X , K , B ):
    return K*((X[:,0]/X[:,1]**0.5)**(-B))

pred_train=predict_USBM(X_train,K,B)
pred_test=predict_USBM(X_test,K,B)


# LISTS TO SAVE METRICS FOR FINAL COMPARISON
R2_values_train=[]
RMSE_values_train=[]
R2_values_test=[]
RMSE_values_test=[]
index=[]

RMSE_values_test.append(MSE(y_test,pred_test)**(1/2))
R2_values_test.append(r2_score(y_test,pred_test))

RMSE_values_train.append(MSE(y_train,pred_train)**(1/2))
R2_values_train.append(r2_score(y_train,pred_train))
index.append("USBM")

plt.figure()
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(pred_test, color = 'blue', label = 'Predicted data')
plt.title('Prediction USBM')
plt.legend()
plt.show()

data={'RMSE Test':RMSE_values_test,'R2 test':R2_values_test,'RMSE_train':RMSE_values_train,"R2_train":R2_values_train}

table=pd.DataFrame(data,index=index)

print(table)

