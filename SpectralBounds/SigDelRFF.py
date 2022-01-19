#-H.K.
#Reference: https://arxiv.org/abs/2106.02614  

import numpy as np
import math
import scipy.linalg
import sys
from numba import jit, njit, cuda, float32

def main():
    
    #Set number of data points
    N = 100
    N_train = 100

    #Set the number of features m
    m = int(sys.argv[1])
    p = int(sys.argv[2])
    lambd = int(m/p)
    mu = 10
    
    #Set sigma for RBF kernel
    sigma = 0.05

    #Generate Data matrix X with shape N x 5
    X,y = GenerateData(N)

    #Compute the RBF Kernel
    K = RBFkernel(sigma,X,N_train)    

    #Compute RFF features and RFF kernel
    Z = RFF_for_RBF(sigma,X,m,N_train)
    K_RFF = (2/m)*np.matmul(Z,Z.transpose())

    #Perform Sigma-Delta quantization to the features
    Q = SigDel(Z,1)

    #Apply condensation operator
    Z_tilde = Condense(Q,1,p)

    #Compute approximate kernel given by Condensed + Sigma-Delta quantization
    K_tilde = np.matmul(Z_tilde,Z_tilde.transpose())

    #Compute spectral bounds for regularized K_RFF
    Delta_1,Delta_2 = SpectralBounds(K,K_RFF,mu)
    print("*******************************************")
    print(str(Delta_1) + "\t" + str(Delta_2))        
    print("*******************************************")

    #Compute spectral bounds for regularized K_tilde
    Delta_1,Delta_2 = SpectralBounds(K,K_tilde,mu)
    print("*******************************************")
    print(str(Delta_1) + "\t" + str(Delta_2))        
    print("*******************************************")

#Function to generate synthetic non-linear data samples
def GenerateData(N,d):    
    X = []
    y = []
    eps_list  = []
    gamma = np.ones(d) 
    for i in range(0,N):
        x_i = np.random.uniform(-1,1,d)
        #eps = 0
        eps = np.random.normal(0,0.25)
        y_i = np.dot(gamma,x_i) + np.dot(gamma,np.cos(np.power(x_i,2))) + np.dot(gamma,np.cos(np.absolute(x_i))) + eps
        X.append(x_i.tolist())
        y.append(y_i)
        eps_list.append(eps)
    X = np.array(X)
    y = np.array(y)
    eps_list = np.array(eps_list)

    return X, y, eps_list

#@jit(nopython=True)   
def RBFkernel(sigma,X,N_train):
    K = np.empty((N_train,N_train))
    for i in range(0,N_train):
        for j in range(0,N_train):
           K[i,j] = math.exp((-0.5/pow(sigma,2))*pow(np.linalg.norm(X[i,:]-X[j,:],2),2))
        
    return K

def RFF_for_RBF(sigma,X,m,N_train,d):
    
    Omega = []
    for i in range(0,m):
        omega_i = np.random.multivariate_normal(np.zeros(d),pow(sigma,-2)*np.identity(d))
        #print(omega_i.shape)
        Omega.append(omega_i.tolist()) 
    Omega = np.array(Omega)
    Omega = np.transpose(Omega)
   
    #print("Shape of Omega matrix is " + str(Omega.shape))

    Z = []
    zeta = np.random.uniform(0,2*math.pi,m)

    for i in range(0,N_train):
        z_i = np.cos(np.matmul(np.transpose(Omega),np.transpose(X[i,:]))+zeta)
        Z.append(z_i.tolist())
    Z = np.array(Z)
    return Z

#Function to perform Sigma-Delta Quantization
def SigDel(Z,r,sig):
    
    nj = np.empty(r, dtype=int)
    for j in range(0,r):
        nj[j] = sig*(j**2)+1
    dj = np.zeros(r)
    for j in range(0,r):
        for i in range(0,r):
            if i != j:
                dj[j] = nj[i]/(nj[i]-nj[j])
    
    #print(dj)
    Q = []
    U = []
    N_train, m = Z.shape
    
    if r == 2:
        for i in range(0,N_train):
            q_i = []
            u_i = []
            u_i.append(0)
            #u_i.append(np.random.uniform(-1,1))
            for k in range(0,m):
                    if k < 6:
                        tmp = dj[0]*u_i[k] + Z[i,k]
                        #Deal with 0
                        if tmp == 0:
                            tmp = 1 
                    else:
                        tmp = dj[0]*u_i[k] + dj[1]*u_i[k-6] + Z[i,k]
                        #Deal with 0
                        if tmp == 0:
                            tmp = 1 
                    q_i.append(np.sign(tmp))
                    u_i.append(tmp -np.sign(tmp))
            Q.append(q_i)
            U.append(u_i)
    elif r == 1:
        for i in range(0,N_train):
            q_i = []
            u_i = []
            u_i.append(0)
            #u_i.append(np.random.uniform(-1,1))

            for k in range(0,m):
                tmp = np.sign(Z[i,k] + u_i[k])
                q_i.append(tmp)
                u_i.append(u_i[k] + Z[i,k] - q_i[k])

            Q.append(q_i)
            U.append(u_i)

    Q = np.array(Q, dtype=int)
    U = np.array(U)
    
    return Q, U

#Function to perform Sigma-Delta Quantization
def GreedySigDel(Z,r,sig,alphabet):

    Q = []
    U = []
    N_train, m = Z.shape
    
    if r == 2:
        for i in range(0,N_train):
            q_i = []
            u_i = []
            u_i.append(np.random.uniform(-1,1))
            
            tmp = Z[i,0] + 2*u_i[0]
            q_i.append(alphabet[np.abs(alphabet-tmp).argmin()])
            u_i.append(tmp - q_i[0])
            
            for k in range(1,m):
                tmp = Z[i,k] + 2*u_i[k] - u_i[k-1]
                q_i.append(alphabet[np.abs(alphabet-tmp).argmin()])
                u_i.append(tmp - q_i[k])

            Q.append(q_i)
            U.append(u_i)         
    elif r == 1:
        for i in range(0,N_train):
            q_i = []
            u_i = []
            u_i.append(np.random.uniform(-1,1))

            for k in range(0,m):
                tmp = Z[i,k] + u_i[k]
                q_i.append(alphabet[np.abs(alphabet-tmp).argmin()])
                u_i.append(u_i[k] + Z[i,k] - q_i[k])

            Q.append(q_i)
            U.append(u_i)

    Q = np.array(Q)
    U = np.array(U)
    
    return Q, U

#Function to apply condensation operator
def Condense(Q,r,p,lambd,v):
    
    N_train, m = Q.shape
    lambd_tilde = int((lambd+r-1)/r)

    if r == 2:
        V_SigDel = np.kron(np.identity(p),v)
        V_SigDel = (math.sqrt(2/p)/np.linalg.norm(v))*V_SigDel
        Z_tilde  = []
        for i in range(0,N_train):
            y_i = np.matmul(V_SigDel,Q[i,:])
            Z_tilde.append(y_i.tolist())
        Z_tilde = np.array(Z_tilde)
    
    elif r == 1:
        V_SigDel = np.kron(np.identity(p),v)
        V_SigDel = (math.sqrt(2/p)/np.linalg.norm(v))*V_SigDel
        Z_tilde  = []
        for i in range(0,N_train):
            y_i = np.matmul(V_SigDel,Q[i,:])
            Z_tilde.append(y_i.tolist())
        Z_tilde = np.array(Z_tilde)
    
    return Z_tilde

def NoiseShapeQuant(Z,lambd,beta):
    N, m = Z.shape
    Q = []
    U = []
    for i in range(0,N):
        
        q_i = []
        u_i = []
        #u_i.append(np.random.uniform(-1/beta,1/beta))
        u_i.append(0)
        
        for k in range(0,m):
            if k%(lambd) == 0:
                #u_i[k] = np.random.uniform(-1/beta,1/beta)
                u_i[k] = 0
                
            q_i.append(np.sign(Z[i,k] + beta*u_i[k]))
            #u_i.append((1/beta)*(beta*u_i[k] + Z[i,k] - q_i[k]))
            u_i.append((beta*u_i[k] + Z[i,k] - q_i[k]))

        Q.append(q_i)
        U.append(u_i)

    Q = np.asarray(Q, dtype=int)
    U = np.asarray(U)
    
    return Q, U

def NoiseShapeCondense(Q,p,lambd,beta):
    N = Q.shape[0]
    
    v = []    
    for i in range(0,lambd):
        v.append(pow(beta,-(i+1)))
        
    #print(v)

    V = np.kron(np.identity(p),v)
    V = (math.sqrt(2/p)/np.linalg.norm(v,2))*V
    
    Z_tilde  = []
    for i in range(0,N):
        y_i = np.matmul(V,Q[i,:])
        Z_tilde.append(y_i.tolist())
    
    Z_tilde = np.asarray(Z_tilde)
    return Z_tilde

#@jit(nopython=True)   
def StocQuant(Q_stoc,Z,m,N_train):
    for i in range(0,N_train):
        for k in range(0,m):
            toss_p = float(0.5*(1-Z[i,k]))
            #p = np.asarray([toss_p,1-toss_p])
            #index = np.searchsorted(np.cumsum(p), np.random.rand(1), side="right")
            #The below in-built option is not compatible with numba. So doing a workaround
            
            Q_stoc[i,k] = np.random.choice([-1,1],p=[toss_p,1-toss_p])
            
            #if index == 0:
            #    Q_stoc[i,k] = -1
            #else:
            #    Q_stoc[i,k] = 1
                
def MultiBitStocQuant(Q_stoc,Z,m,N_train,alphabet):
    for i in range(0,N_train):
        for k in range(0,m):
            
            for j in range(0,len(alphabet)-1):
                if alphabet[j] <= Z[i,k] and Z[i,k] <= alphabet[j+1]:
                    s = alphabet[j]
                    t = alphabet[j+1]
            
            toss_p = float((t-Z[i,k])/(t-s))
            Q_stoc[i,k] = np.random.choice([s,t],1,p=[toss_p,1-toss_p])
            
#Functions to refine the bounds Delta_1 and Delta_2
#@jit(nopython=True)   
def refine_delta1(K,K_apx,mu,Delta_1):
    low = 0
    upper = Delta_1
    
    N_train, N_train = K.shape
    I = np.identity(N_train)
    
    mid = (low+upper)/2
    temp_eig = np.linalg.eigvalsh(K_apx+mu*I - (1-mid)*(K + mu*I))
  
    while ((temp_eig < 0).any() or upper-low > 1e-8):

        if (temp_eig>0).all():
            upper = mid
        else:
            low = mid        
               
        mid = (low+upper)/2
        temp_eig = np.linalg.eigvalsh(K_apx+mu*I - (1-mid)*(K + mu*I))
       
        #print(mid)
        #print(temp_eig)
        #print("***********")
        
    return mid

#@jit(nopython=True)   
def refine_delta2(K,K_apx,mu,Delta_2):
    low = 0
    upper = Delta_2
    
    N_train, N_train = K.shape
    I = np.identity(N_train)
    
    mid = (low+upper)/2
    temp_eig = np.linalg.eigvalsh((1+mid)*(K + mu*I) - (K_apx+mu*I))
    
    while ((temp_eig < 0).any() or upper-low > 1e-8):

        if (temp_eig>0).all():
            upper = mid
        else:
            low = mid        
               
        mid = (low+upper)/2
        temp_eig = np.linalg.eigvalsh((1+mid)*(K + mu*I) - (K_apx+mu*I))
        
        #print(mid)
        #print(temp_eig)
        #print("***********")
        
    return mid

def SpectralBounds(K,K_apx,mu):

    N_train, N_train = K.shape
    I = np.identity(N_train)
    
    Eig_valuesA = np.linalg.eigvalsh(K+mu*I)
    max_true = np.amax(Eig_valuesA)
    min_true = np.amin(Eig_valuesA)

    Eig_valuesB = np.linalg.eigvalsh(K_apx+mu*I)
    max_tilde = np.amax(Eig_valuesB)
    min_tilde = np.amin(Eig_valuesB)
    
    Delta_1 = 1 - (min_tilde/max_true)
    Delta_2 = (max_tilde/min_true) -1

    if Delta_1 > 0:
        Delta_1 = refine_delta1(K,K_apx,mu,Delta_1)
    else:
        Delta_1 = 0
        print(Delta_1)
    
    if Delta_2 > 0:
        Delta_2 = refine_delta2(K,K_apx,mu,Delta_2)
    else:
        Delta_2 = 0
    
    return Delta_1,Delta_2

def checkBounds(Delta_1,Delta_2,K,K_apx,mu):
    N_train, N_train = K.shape
    I = np.identity(N_train)

    temp_eig1 = np.linalg.eigvalsh(K_apx + mu*I - (1-Delta_1)*(K + mu*I))
    temp_eig2 = np.linalg.eigvalsh((1+Delta_2)*(K + mu*I) - (K_apx+mu*I))

    if (temp_eig1 < 0).any() or (temp_eig2 < 0).any():
        print("Bounds INVALID")

def CleanSpectralBounds(K,K_apx,mu):

    N_train, N_train = K.shape
    I = np.identity(N_train)

    tmp,err = scipy.linalg.sqrtm(K + mu*I,disp=False)
    tmp = np.linalg.inv(tmp)

    eigs = np.linalg.eigvals(np.matmul(tmp,np.matmul(K-K_apx,tmp)))
    
    #Check this bruv
    Delta_2 = -np.amin(eigs)
    Delta_1 =  np.amax(eigs)

    return Delta_1, Delta_2
if __name__== "__main__":
    main()
