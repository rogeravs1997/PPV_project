import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import r2_score

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
def USBM_transform(X,y):
    """
    Tansform a dataset into USBM format.

    Parameters
    ----------
    X (numpy.ndarray) : The independent variable 
    y (numpy.ndarray) : The dependent data

    Returns
    -------
    X_transformed (numpy.ndarray) : The independent variable transformed 
    y_transformed (numpy.ndarray) : The dependent data transformed

    """
    X_transformed=np.log(X[:,0]/np.sqrt(X[:,1]))
    y_transformed=np.log(y)
    X_transformed.shape, y_transformed.shape = (-1, 1), (-1, 1)
    
    return X_transformed , y_transformed
    
X_train_transform , y_train_transform = USBM_transform(X_train, y_train)


#add an extra column of 1's to X
X_train_transform=np.concatenate((np.ones((len(y_train_transform),1)),X_train_transform),axis=1)


#defining function to compute the Cost of J(theta)
def computeCost(X,y,theta):
    """
    Compute a variant of mean square error.
    
    Parameters
    ----------
    X (numpy.ndarray) : The independent variable 
    y (numpy.ndarray) : The dependent data
    theta (numpy.ndarray) : Parameters

    Returns
    -------
    J (float) : Cost

    """
    m=len(y)
    pred=X.dot(theta)
    sqrError=np.square(pred-y)
    J=(1/(2*m))*np.sum(sqrError)
    return J

def gradientDescent(X, y, theta, alpha=0.01, num_iters=100):
    """
    Gradient descent algorithm for linear models
    
    Parameters
    ----------
    X (numpy.ndarray) : The independent variable
    y (numpy.ndarray) : The dependent data
    theta (numpy.ndarray) : Parameters to optimize
    alpha (float) : Learning rate. The default is 0.01.
    num_iters (int) : Number of iterations. The default is 100.

    Returns
    -------
    theta (numpy.ndarray) : Optimized parameters
    J (list) : Cost history

    """
    m=len(y)
    J=[]
    for i in range(num_iters):
        error=X.dot(theta)-y
        delta=(1/m)*error.T.dot(X)
        theta=theta-(alpha*delta.T)
        J.append([computeCost(X,y,theta)])
    J=np.array(J)
    return theta , J



#setting an initial values for theta , alpha and # of iters
initial_theta=np.ones((len(X_train_transform[1]),1))
alpha=0.25
num_iters=500 

#dumping values of theta in theta_found and the history o costJ in J_history 
theta_found,J_history=gradientDescent(X_train_transform,y_train_transform,initial_theta,alpha,num_iters)

# getting values of K and B
K=np.exp(theta_found[0])
B=-1*theta_found[1]


#plotting cost of J in function of number of iterations
def plotCost(J_history):
    fig = plt.figure()
    plt.plot(J_history)
    fig.suptitle('Cost(J) in the time', fontsize=15)
    plt.xlabel('Number of Iterations', fontsize=8)
    plt.ylabel('Cost(J)', fontsize=8)
    
plotCost(J_history)


def predict_USBM ( X , K , B ):
    """
    Predict values of Peak Particle Velocity given a dataset and constants K and B

    Parameters
    ----------
    X (numpy.ndarray) : Dependant variables. Where: X[:,0] = Distance from measure point to blast site and
                                                      X[:,1] = Maximun Charge per Delay
    K (float) : Site constant
    B (float) : Site constant

    Returns
    -------
    pred (numpy.ndarray) : Peak Particle Velocity Predictions

    """
    pred=K*((X[:,0]/X[:,1]**0.5)**(-B))
    return pred

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

