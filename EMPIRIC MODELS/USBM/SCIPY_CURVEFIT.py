
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# LOADING THE DATA FROM A .MAT FILE (MATLAB/OCTAVE FILE)
file_name = "limpiado"
main_path = ("D:\Desktop\Modelo Predictivo PPV\database")
file_path = (file_name + ".xlsx")
sheet_name = "data"
dataset = pd.read_excel(main_path + "\\" + file_path, sheet_name)
dataset=dataset.sample(n=200,random_state=1)

# DEFINING FEATURES AND LABELS

X=dataset[["Distancia (m)", "Kg de columna explosiva"]]


y=dataset["PPV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3,shuffle=True)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)



# LISTS TO SAVE METRICS FOR FINAL COMPARISON
R2_values_train=[]
RMSE_values_train=[]
R2_values_test=[]
RMSE_values_test=[]
index=[]


K_values=[]
B_values=[]
A_values=[]
alpha_values=[]
index_2=[]

######################## USBM DUVALL ########################
#DEFINING FUNCTION

def function_1 ( X_train , K , B ):
    return np.log(K)-(B*np.log(X_train[:,0]/X_train[:,1]**0.5))

#FITIING THE DATA
popt, pcov = curve_fit(function_1, X_train, np.log(y_train))

K=popt[0]
B=popt[1]

#EVALUATING THE FIT
        

def predict_USBM ( X , K , B ):
    return K*((X[:,0]/X[:,1]**0.5)**(-B))

pred_train=predict_USBM(X_train,K,B)
pred_test=predict_USBM(X_test,K,B)

# SAVING METRICS
RMSE_values_test.append(MSE(y_test,pred_test)**(1/2))
R2_values_test.append(r2_score(y_test,pred_test))

RMSE_values_train.append(MSE(y_train,pred_train)**(1/2))
R2_values_train.append(r2_score(y_train,pred_train))
index.append("USBM")

# SAVING PARAMETERS
K_values.append(K)
B_values.append(B)
A_values.append("-")
alpha_values.append("-")
index_2.append("USBM")

#PLOTTING PREDICTIONS VS REAL (ON TEST SET)
plt.plot(y_train, color = 'red', label = 'Real data')
plt.plot(pred_train, color = 'blue', label = 'Predicted data')
plt.title('Prediction USBM')
plt.legend()
plt.show()

data={'RMSE Test':RMSE_values_test,'R2 test':R2_values_test,'RMSE_train':RMSE_values_train,"R2_train":R2_values_train}

table=pd.DataFrame(data,index=index)

print(table)
