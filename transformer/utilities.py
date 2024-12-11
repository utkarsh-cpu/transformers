import numpy as np
import torch as th

def one_hot_it(z):
    s=np.zeros((z.shape[0],int(np.amax(z)+1)))
    for i in range(z.shape[0]):
        s[i,z[i][0]]=1
    return s

def logistic_function(A,b):
    X=1/(1+np.exp(-b*A))
    return X

def d_logistic(A,b):
    X=b*A*(1-A)
    return X

def tanh(A,b):
    X=(np.exp(b*A)-np.exp(-1*b*A))/(np.exp(b*A)+np.exp(-1*b*A))
    return X

def d_tanh(A,b):
    X=b*(1-A**2)
    return X

def loss_and_grad_perceptron(A,X,Y,b,d_activation_function=d_logistic):
    loss=np.mean(0.5*(Y-A)**2,axis=0)
    grad_w=np.dot(X.T,(-(Y-A)*d_activation_function(A,b)))
    return loss,grad_w
    
def loss_and_grad_MLFFNN(A_temp,Y,b,learning_rate,weights,biases,d_activation_function=d_logistic):
    '''
    print(len(A_temp))
    print("A")
    for i in A_temp:
        print(i.shape)
    
    print("W")
    for i in weights:
        print(i.shape)
    print("B")
    for i in biases:
        print(i.shape)
    '''
    if d_activation_function==d_regression:
        loss=np.mean(0.5*(Y-A_temp[-1])**2,axis=0)
    else:
        loss=-np.sum(Y*np.log(A_temp[-1]))/Y.shape[0]
    new_weights,new_biases=back_prop(A_temp,Y,b,learning_rate,weights,biases,d_activation_function)
    return loss,new_weights,new_biases

def back_prop(A_temp,Y,b,learning_rate,weights,biases,d_activation_function):
    new_weights=weights[:]
    new_biases=biases[:]
    grad_w=(np.dot(A_temp[-2].T,(Y-A_temp[-1])))      #(-1/Y.shape[0])*
    #print(grad_w.shape)
    new_weights[-1]=new_weights[-1]-learning_rate*grad_w
    grad_A=(np.dot((Y-A_temp[-1]),weights[-1].T))
    #print(grad_A.shape)
    grad_b=(np.dot(np.ones((1,A_temp[-2].shape[0])),(Y-A_temp[-1])))
    #print(grad_b.shape)
    new_biases[-1]=new_biases[-1]-learning_rate*grad_b
    for i in range(1,len(weights)):
        grad_w=np.dot(A_temp[-i-2].T,grad_A*d_activation_function(A_temp[-i-1],b))
        #print(grad_w.shape)
        new_weights[-i-1]=new_weights[-1-i]-learning_rate*grad_w
        grad_b=np.dot(np.ones((1,A_temp[-i-2].shape[0])),grad_A*d_activation_function(A_temp[-i-1],b))
        #print(grad_b.shape)
        new_biases[-i-1]=new_biases[-1-i]-learning_rate*grad_b
        grad_A=np.dot(grad_A*d_activation_function(A_temp[-i-1],b),weights[-i-1].T)
        #print(grad_A.shape)
    return new_weights,new_biases

def regression(A,b):
    return A

def d_regression(A,b):
    return 1

def relu(x,b):
    s=b*x*(x>0)
    return s

def drelu(x,b):
    s=b*(x>0)
    return s

def softmax(A):
    A-=np.max(A)
    A_exp=np.exp(A)
    A_sum=(np.sum(A_exp,axis=0)).reshape((1,A.shape[1]))
    s=A_exp/A_sum
    return s

def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = th.arange((maxlen), dtype=th.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = th.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return softmax(X.reshape(shape))

'''
a=np.array([[1,1,1,1],[2,3,4,1]])
y=np.array([[1,0]])
y=one_hot_it(y)
w1=np.array([1,1,1,1])
w2=np.array([2,1,2,4])
w3=np.array([2,3,1,5])
A1=np.dot(a,w1)
A2=np.dot(a,w2)
A3=np.dot(a,w3)
A=np.array([A1,A2,A3])
print(A)

loss,grad=loss_and_grad_perceptron(A,a,y,1)
print(loss)
print(grad)
learning_rate=1
w1=w1.astype("float64")
w2=w2.astype("float64")
w3=w3.astype("float64")
w1+=learning_rate*grad[0,:]
w2+=learning_rate*grad[1,:]
w3+=learning_rate*grad[2,:]
print(w1,w2,w3)
'''

'''
(33, 24) W0
(120, 25) A0
(25, 12) W1
(120, 13) A1
(13, 6) W2
(120, 7) A2
(7, 3) W3
(120, 3) A3
(120, 3)
'''
