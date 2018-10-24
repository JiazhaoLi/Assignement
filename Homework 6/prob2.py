
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(17)

def initialize(K):
    Initial_A_2 = np.array([
        [0.4,0.6],  # S_0 ..S_!
        [0.6,0.4]
    ])
    # emission matrix
    Initial_phi_2 = np.array([
        [0.5, 0.1, 0.2, 0.2],#S_1
        [0.1, 0.5, 0.1, 0.3] #s_2
    ])   #A   . C  . G  . T
    # for K = 4 , you should use the following parameters to initialize
    # transition matrix
    Initial_A_4 = np.array([
        [0.3, 0.1, 0.2, 0.4],
        [0.1, 0.2, 0.4, 0.3],
        [0.2, 0.4, 0.3, 0.1],
        [0.4, 0.3, 0.1, 0.2]]
    )
    # emission matrix
    Initial_phi_4 = np.array([
        [0.5, 0.1, 0.2, 0.2],
        [0.1, 0.5, 0.1, 0.3],
        [0.1, 0.2, 0.5, 0.2],
        [0.3, 0.1, 0.1, 0.5]
    ])
    if K == 2:
        return Initial_A_2,Initial_phi_2
    if K == 4:
        return Initial_A_4,Initial_phi_4
    
def forward_backward(O_1,O_2,V,Q,A,B):
    # T is len of observation,(10/9),V output vocabulary(4 ACGT), hidden stat set Q(2/4)
    for it in range(10):
    #get initial state:
        if Q == 2:
            initial_state = [0.5,0.5]
        if Q == 4:
            initial_state = [0.25,0.25,0.25,0.25]
        L_1 = len(O_1)
        L_2 = len(O_2)

        #transfer the ob into number in array
        O_1_list = []
        O_2_list = []
        for i in range(L_1):
            if O_1[i] == "A":
                O_1_list.append(0)
            elif O_1[i] == "C":
                O_1_list.append(1)
            elif O_1[i] == "G":
                O_1_list.append(2)
            elif O_1[i] == "T":
                O_1_list.append(3)
        for i in range(L_2):
            if O_2[i] == "A":
                O_2_list.append(0)
            elif O_2[i] == "C":
                O_2_list.append(1)
            elif O_2[i] == "G":
                O_2_list.append(2)
            elif O_2[i] == "T":
                O_2_list.append(3)
        

        #calculate the alpha
        alpha_1 = np.zeros((Q,L_1))
        alpha_2 = np.zeros((Q,L_2))
        
        for i in range(Q):
            alpha_1[i,0] = initial_state[i]*B[i,O_1_list[0]]
        #initialize the first alpha
        for i in range(1,L_1): # time i
            for j in range(Q):#state j
                for k in range(Q): #sum
                    alpha_1[j,i] += alpha_1[k,i-1]*A[k,j]*B[j,O_1_list[i]]
            
        alpha1_T= 0  
        for i in range(Q):
            alpha1_T += alpha_1[i,L_1-1]   
#         alpha1_T = alpha_1[0,L_1-1]+alpha_1[1,L_1-1]

#         alpha_2[0,0] = initial_state[0]*B[0,O_2_list[0]]
#         alpha_2[1,0] = initial_state[1]*B[1,O_2_list[0]]
        for i in range(Q):
            alpha_2[i,0] = initial_state[i]*B[i,O_2_list[0]]
            
        for i in range(1,L_2): # time i
            for j in range(Q):#state j
                for k in range(Q): #sum
                    alpha_2[j,i] += alpha_2[k,i-1]*A[k,j]*B[j,O_2_list[i]] 
        alpha2_T = 0
        for i in range(Q):
            alpha2_T += alpha_2[i,L_2-1]



        #calculate the beta
        beta_1 = np.zeros((Q,L_1))
        beta_2 = np.zeros((Q,L_2))

        for i in range(Q):
            beta_1[i,L_1-1] = 1
#         beta_1[1,L_1-1] = 1
        for i in range(L_1-2,-1,-1):
            for j in range(Q):
                for k in range(Q):
                    beta_1[j,i] += beta_1[k,i+1]*A[j,k]*B[k,O_1_list[i+1]]
        beta1_0 = np.zeros(np.shape(beta_1))
        for i in range(Q):
            beta1_0 += B[i,O_1_list[0]]*beta_1[i,0]

        for i in range(Q):
            beta_2[i,L_2-1] = 1
        for i in range(L_2-2,-1,-1):
            for j in range(Q):
                for k in range(Q):
                    beta_2[j,i] += beta_2[k,i+1]*A[j,k]*B[k,O_2_list[i+1]]
        beta2_0 = np.zeros(np.shape(beta_2))
        for i in range(Q):
            beta2_0+= B[i,O_2_list[0]]*beta_2[i,0]

#         beta2_0 = B[0,O_2_list[0]]*beta_2[0,0]+B[1,O_2_list[0]]*beta_2[1,0]


        #E_step:
        
        gamma_1 = alpha_1*beta_1/alpha1_T
        gamma_2 = alpha_2*beta_2/alpha2_T

        xi_1 = np.zeros((Q,Q,L_1))
        for t in range(L_1-1): #t
            for i in range(Q):   # i
                for j in range(Q):  # j
                    xi_1[i,j,t] = alpha_1[i,t]*A[i,j]*beta_1[j,t+1]*B[j,O_1_list[t+1]] /alpha1_T
        xi_2 = np.zeros((Q,Q,L_2))
        for t in range(L_2-1): #t
            for i in range(Q):   # i
                for j in range(Q):  # j
                    xi_2[i,j,t] = alpha_2[i,t]*A[i,j]*beta_2[j,t+1]*B[j,O_2_list[t+1]] / alpha2_T

        #M _step:

        for i in range(np.shape(A)[0]):
            for j in range(np.shape(A)[1]):
                A[i,j] = (sum(xi_1[i,j,:])+sum(xi_2[i,j,:])) / (sum(sum(xi_1[i,:,:]))+sum(sum(xi_2[i,:,:])))
        B = np.zeros(np.shape(B))
        for i in range(Q): # state
            for j in range(V): # for vocabularty
                for t in range(L_1):
                    if O_1_list[t] == j:
                        B[i,j] += gamma_1[i,t]
                for t in range(L_2):
                    if O_2_list[t] == j:
                        B[i,j] += gamma_2[i,t]

    #         print((sum(gamma_1[i,:])+sum(gamma_2[i,:])))
    #         print(B)
    #         print(B)
        for i in range(Q):
            for j in range(V):
                B[i,j] = B[i,j] / (sum(gamma_1[i,:])+sum(gamma_2[i,:]))

    
    return A,B

ob_1 = ["C","C","T","A","C","A","C","G","C","A"]
len(ob_1)
ob_2 = ["C","T","A","C","G","C","A","A","T"]
len(ob_2)
Initial_A_2,Initial_phi_2 = initialize(2)
Initial_A_4,Initial_phi_4 = initialize(4)
#A= forward_backward(ob_1,ob_2,4,2,Initial_A_2,Initial_phi_2)
A= forward_backward(ob_1,ob_2,4,2,Initial_A_2,Initial_phi_2)
B= forward_backward(ob_1,ob_2,4,4,Initial_A_4,Initial_phi_4)
print(A)
print(B)  


# In[393]:




