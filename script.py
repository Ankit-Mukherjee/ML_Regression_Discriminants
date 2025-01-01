import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def transposeOfAMatrix(A):
    return A.T
    
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    unique_classes = np.unique(y)
    means = np.zeros((X.shape[1], len(unique_classes)))
    for idx, label in enumerate(unique_classes):
        means[:, idx] = np.mean(X[y.flatten() == label], axis=0)
    
    covmat = np.cov(transposeOfAMatrix(X))
    return means, covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    unique_classes = np.unique(y)
    means = np.zeros((X.shape[1], len(unique_classes)))
    covmats = []
    
    for idx, label in enumerate(unique_classes):
        means[:, idx] = np.mean(X[y.flatten() == label], axis=0)
        covmats.append(np.cov(X[y.flatten() == label].T))
    
    return means, covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    inv_cov = np.linalg.inv(covmat)
    ypreds = []
    
    for x in Xtest:
        distances = []
        for mean in means.T:
            distance = (x - mean).T @ inv_cov @ (x - mean)
            distances.append(distance)
        ypreds.append(np.argmin(distances) + 1)
    
    ypreds = np.array(ypreds)
    acc = np.mean(ypreds == ytest.flatten()) 
    return acc,ypreds

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ypred = []
    
    for x in Xtest:
        distances = []
        for idx, mean in enumerate(means.T):
            cov_inv = np.linalg.inv(covmats[idx])
            distance = (x - mean).T @ cov_inv @ (x - mean)
            distances.append(distance)
        ypred.append(np.argmin(distances) + 1)
    
    ypred = np.array(ypred)
    acc = np.mean(ypred == ytest.flatten()) 
    return acc,ypred

def learnOLERegression(X,y):
    # Compute the optimal weights using the formula: w = (X^T X)^-1 X^T y
    weight = np.linalg.inv(transposeOfAMatrix(X) @ X) @ transposeOfAMatrix(X) @ y
    return weight

def learnRidgeRegression(XMatrix,yMatrix,lambdaa):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    Inversion = XMatrix.shape[1]
    weight = np.linalg.inv(transposeOfAMatrix(XMatrix) @ XMatrix + lambdaa * np.eye(Inversion)) @ transposeOfAMatrix(XMatrix) @ yMatrix
    return weight

def testOLERegression(weight,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    y_pred = Xtest @ weight
    
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((ytest - y_pred) ** 2)
    return mse

def regressionObjVal(weight, XMatrix, yMatrix, lambdaa):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    
    # Compute predictions: X is the input matrix, w is the weight vector
    predictions = np.dot(XMatrix, weight)
    
    # Reshape y to ensure it's a column vector
    yMatrix = yMatrix.reshape(-1)  # Make sure y is a 1D array of shape (65,)
    
    # Compute the error (squared loss)
    error = (1 / 2) * np.sum((predictions - yMatrix) ** 2) + (lambdaa / 2) * np.sum(weight ** 2)  # Regularization term

    # Compute the gradient of the error with respect to w
    error_grad = np.dot(transposeOfAMatrix(XMatrix), (predictions - yMatrix)) + lambdaa * weight
    return error,error_grad

def mapNonLinear(xMatrix,prompt):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    xMatrix = xMatrix.reshape(-1, 1)
    Xp = np.ones((xMatrix.shape[0], prompt + 1))
    for i in range(1, prompt + 1):
        Xp[:, i] = (xMatrix ** i).ravel()
    return Xp

# Main script
if __name__ == "__main__":
    # Problem 1
    # load the sample data                                                                 
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

    # LDA
    means,covmat = ldaLearn(X,y)
    ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
    print('LDA Accuracy = '+str(ldaacc))
    # QDA
    means,covmats = qdaLearn(X,y)
    qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
    print('QDA Accuracy = '+str(qdaacc))

    # plotting boundaries
    x1 = np.linspace(-5,20,100)
    x2 = np.linspace(-5,20,100)
    xx1,xx2 = np.meshgrid(x1,x2)
    xx = np.zeros((x1.shape[0]*x2.shape[0],2))
    xx[:,0] = xx1.ravel()
    xx[:,1] = xx2.ravel()

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)

    zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
    plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
    plt.title('LDA')

    plt.subplot(1, 2, 2)

    zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
    plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
    plt.title('QDA')

    plt.show()
    # Problem 2
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

    # add intercept
    X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

    w = learnOLERegression(X,y)
    #w=w.flatten()
    mle = testOLERegression(w,Xtest,ytest)
    #print(Xtest.shape," ",w.shape)
    w_i = learnOLERegression(X_i,y)
    #w_i=w_i.flatten()
    mle_i = testOLERegression(w_i,Xtest_i,ytest)
    #print(Xtest_i.shape," ",w_i.shape)
    print('MSE without intercept '+str(mle))
    print('MSE with intercept '+str(mle_i))
    
    #predictions_without_intercept = X @ w
    #predictions_with_intercept = X_i @ w_i

    # Problem 3
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses3_train = np.zeros((k,1))
    mses3 = np.zeros((k,1))
    for lambd in lambdas:
        w_l = learnRidgeRegression(X_i,y,lambd)
        mses3_train[i] = testOLERegression(w_l,X_i,y)
        mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')

    plt.show()
    # Problem 4
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses4_train = np.zeros((k,1))
    mses4 = np.zeros((k,1))
    opts = {'maxiter' : 20}    # Preferred value.                                                
    w_init = np.ones((X_i.shape[1],1)).flatten()
    for lambd in lambdas:
        args = (X_i, y, lambd)
        w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
        w_l = np.transpose(np.array(w_l.x))
        w_l = np.reshape(w_l,[len(w_l),1])
        mses4_train[i] = testOLERegression(w_l,X_i,y)
        mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses4_train)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.legend(['Using scipy.minimize','Direct minimization'])

    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses4)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')
    plt.legend(['Using scipy.minimize','Direct minimization'])
    plt.show()


    # Problem 5
    pmax = 7
    lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
    mses5_train = np.zeros((pmax,2))
    mses5 = np.zeros((pmax,2))
    for p in range(pmax):
        Xd = mapNonLinear(X[:,2],p)
        Xdtest = mapNonLinear(Xtest[:,2],p)
        w_d1 = learnRidgeRegression(Xd,y,0)
        mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
        mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
        w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
        mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
        mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(range(pmax),mses5_train)
    plt.title('MSE for Train Data')
    plt.legend(('No Regularization','Regularization'))
    plt.subplot(1, 2, 2)
    plt.plot(range(pmax),mses5)
    plt.title('MSE for Test Data')
    plt.legend(('No Regularization','Regularization'))
    plt.show()