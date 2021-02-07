
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

#DEFINING FUNCTION

def function_4( X_train , K , B , alpha):
    return K*((X_train[:,0]/X_train[:,1]**(1/3))**(-B))*np.exp(-alpha*X_train[:,0])

#FITIING THE DATA
popt, pcov = curve_fit(function_4, X_train, y_train,p0=[400,1,0.001])

K_4=popt[0]
B_4=popt[1]
alpha_4=popt[2]

#EVALUATING THE FIT
        
pred_test_4=function_4(X_test,K_4,B_4,alpha_4)
pred_train_4=function_4(X_train,K_4,B_4,alpha_4)

# SAVING METRICS
RMSE_values_test.append(MSE(y_test,pred_test_4)**(1/2))
R2_values_test.append(r2_score(y_test,pred_test_4))

RMSE_values_train.append(MSE(y_train,pred_train_4)**(1/2))
R2_values_train.append(r2_score(y_train,pred_train_4))
index.append("Ghosh Daemen")


# SAVING PARAMETERS
K_values.append(K_4)
B_values.append(B_4)
A_values.append("-")
alpha_values.append(alpha_4)
index_2.append("Ghosh Daemen")

#PLOTTING PREDICTIONS VS REAL (ON TEST SET)
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(pred_test_4, color = 'blue', label = 'Predicted data')
plt.title('Ghosh Daemen')
plt.legend()
plt.show()



# PRINTING FINAL RESULTS TABLE
data={'RMSE Test':RMSE_values_test,'R2 test':R2_values_test,'RMSE_train':RMSE_values_train,"R2_train":R2_values_train}
table=pd.DataFrame(data,index=index)
print(table)

# PRINTING PARAMETERS TABLE
data_2={'K':K_values,'B':B_values,'A':A_values,"alpha":alpha_values}
table_2=pd.DataFrame(data_2,index=index_2)
print(table_2)