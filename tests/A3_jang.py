#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Cars.csv')


# In[3]:


df.shape


# In[4]:


df1 = df.replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5])


# In[5]:


df2 = df1[df1["fuel"].str.contains('CNG|LPG')==False]


# In[6]:


df2['mileage'] = df2['mileage'].str.extract(r'(\d+\.\d+)').astype(float)


# In[7]:


df2['engine'] = df2['engine'].str.extract(r'(\d+)').astype(float)
df2['max_power'] = df2['max_power'].str.extract(r'(\d+\.?\d+)').astype(float)


# In[8]:


df3 = df2.drop(['torque'], axis=1)


# In[9]:


df3["name"] = df3["name"].str.split().str[0]


# In[10]:


df3.drop(df3[df3['owner'] == 5].index, inplace= True)

# In[160]:

df3['selling_price_log'] = np.log(df3['selling_price'])


# In[223]:


bins = [10.11, 11.61, 13.11, 14.61, df3['selling_price_log'].max()]
labels = [0, 1, 2 ,3]

df3['price_category'] = pd.cut(df3['selling_price_log'], bins=bins, labels=labels, include_lowest=True)


# In[224]:


print(df3[['selling_price_log', 'price_category']].head())


# In[192]:


df3.head()


# In[225]:


X_1 = df3[['year','max_power' , 'engine','km_driven']]
y_1 = df3["price_category"]


# In[226]:


X_1.head()
y_1.head()


# In[227]:


from sklearn.model_selection import train_test_split

X_1train, X_1test, y_1train, y_1test = train_test_split(X_1, y_1, test_size = 0.3, random_state = 32)


# In[228]:


X_1train['year'].fillna(X_1train['year'].median(), inplace=True)
X_1train['max_power'].fillna(X_1train['max_power'].median(), inplace=True)
#X_1train['mileage'].fillna(X_1train['mileage'].median(), inplace=True)
X_1train['engine'].fillna(X_1train['engine'].median(), inplace=True)
X_1train['km_driven'].fillna(X_1train['km_driven'].median(), inplace=True)
#X_1train['seats'].fillna(X_1train['seats'].median(), inplace=True)
X_1test['year'].fillna(X_1test['year'].median(), inplace=True)
X_1test['max_power'].fillna(X_1test['max_power'].median(), inplace=True)
#X_1test['mileage'].fillna(X_1test['mileage'].median(), inplace=True)
X_1test['engine'].fillna(X_1test['engine'].median(), inplace=True)
X_1test['km_driven'].fillna(X_1test['km_driven'].median(), inplace=True)
#X_1test['seats'].fillna(X_1test['seats'].median(), inplace=True)



# In[229]:


X_1train.head(20)


# In[198]:


print(X_1train.shape)
print(y_1train.shape)
print(X_1test.shape)
print(y_1test.shape)


# In[174]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time

#Step 1: Prepare data


# In[230]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_1train)
#X_test = scaler.transform(y_1test)


# In[231]:


intercept = np.ones((X_train.shape[0], 1))
X_train   = np.concatenate((intercept, X_train), axis=1)  #add intercept
intercept = np.ones((X_1test.shape[0], 1))  # สร้างคอลัมน์ที่มีค่า 1
X_1test = np.concatenate((intercept, X_1test), axis=1)


# In[ ]:





# In[201]:


y_1train.shape


# In[232]:


k = len(set(y_1train))  # จำนวนคลาส
m = X_train.shape[0]  # จำนวนตัวอย่าง
n = X_train.shape[1]  # จำนวนฟีเจอร์

Y_train_encoded = np.zeros((m, k))  # สร้างอาเรย์ขนาด (m, k)

for each_class in range(k):
    cond = (y_1train == each_class)  # Boolean mask
    Y_train_encoded[cond, each_class] = 1  # กำหนดค่าที่ตรงกันเป็น 1


# In[233]:


k = len(set(y_1test))
m = X_1test.shape[0]  # จำนวนตัวอย่าง
n = X_1test.shape[1]
Y_test_encoded = np.zeros((m, k))  # สร้างอาเรย์ขนาด (m, k)


for each_class in range(k):
    cond = (y_1test == each_class)  # Boolean mask
    Y_test_encoded[cond, each_class] = 1  # กำหนดค่าที่ตรงกันเป็น 1


# In[234]:


class LogisticRegression:
    
    def __init__(self, k, n, method, alpha = 0.001, max_iter=50000 , lambda_=0.0, use_penalty=False):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.lambda_ = lambda_
        self.use_penalty = use_penalty 
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 2000 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y*np.log(h)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)

        if self.use_penalty:
            reg_term = self.lambda_ * self.W
            reg_term[0, :] = 0
            loss += self.lambda_ * np.sum(np.square(self.W[1:, :])) / 2
            grad += reg_term
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()


# In[235]:


class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)


class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta

class Ridge(LogisticRegression):
    
    def __init__(self, method, lr, l):
        self.regularization = RidgePenalty(l)


# In[236]:


model = LogisticRegression(k, X_train.shape[1], "minibatch")

model.fit(X_train, Y_train_encoded)
#print("Shape of X_train:", X_train.shape)  # ควรเป็น (2409, n_features)
#print("Shape of W:", model.W.shape)  # ควรเป็น (n_features, k)
#print("Shape of X_1test:", X_1test.shape)
yhat = model.predict(X_1test)

#model.plot()

#print("=========Classification report=======")
#print("Report: ", classification_report(y_1test, yhat))


# In[238]:


accuracy = np.sum(yhat == y_1test)/ len(y_1test)
num_classes = len(np.unique(y_1test))# จำนวนคลาสทั้งหมด
precision_per_class = []
recall_per_class = []
f1_per_class = []
for c in range(num_classes): #วนเช็คคลาสทั้งหมด
    TP_c = np.sum((yhat == c) & (y_1test == c))#ถ้าหา C = 1 ใน yhat และเ y_test ก็เป็น 1 
    FP_c = np.sum((yhat== c) & (y_1test != c)) # ถ้าหา C = 1 ใน yhat และ y_test ไม่เป็น 1 
    FN_c = np.sum((yhat != c) & (y_1test == c))
    
    precision_c = TP_c / (TP_c + FP_c) if (TP_c + FP_c) > 0 else 0
    recall = TP_c/(TP_c + FN_c) if (TP_c + FN_c) > 0 else 0
    f1 = (2*precision_c*recall)/(precision_c + recall) if (precision_c + recall) > 0 else 0
    precision_per_class.append(precision_c)
    recall_per_class.append(recall)
    f1_per_class.append(f1)
    
print("Precision for each class:", precision_per_class)
print("recall : " , recall_per_class)   
print("F1 : ", f1_per_class)
print("accuracy : ", accuracy)

macro_precision = sum(precision_per_class)/4
macro_recall = sum(recall_per_class)/4
macro_F1 = sum(f1_per_class)/4

print("macro_precision : ", '{:.2f}'.format(macro_precision))
print("macro_recall : ", '{:.2f}'.format(macro_recall))
print("macro_F1 : ", '{:.2f}'.format(macro_F1))


# In[239]:


W = np.array([0.2, 0.3, 0.2, 0.3])
weighted_precision = np.sum(W*precision_per_class)
weighted_recall = np.sum(W*recall_per_class)
weighted_F1 = np.sum(W*f1_per_class)

print("weighted_precision:", '{:.2f}'.format(weighted_precision))
print("weighted_recall : " , '{:.2f}'.format(weighted_recall))   
print("Fweighted_F1 : ", '{:.2f}'.format(weighted_F1))


# 

# scikit-learn

# In[240]:


print("=========Classification report=======")
print("Report: ", classification_report(y_1test, yhat))


# In[241]:


class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)


class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta

class Ridge(LogisticRegression):
    def __init__(self, k, n, method, alpha=0.001, max_iter=50000, lambda_=0.0, use_penalty=True):
        super().__init__(k, n, method, alpha, max_iter, lambda_, use_penalty)


# In[242]:


print(X_train.shape)


# In[244]:


model = LogisticRegression(k=4, n=X_train.shape[1] , method="batch", alpha=0.01, max_iter=10000, lambda_=0.6, use_penalty=True)
model.fit(X_train, Y_train_encoded)
preds = model.predict(X_1test)
model.plot()


# In[245]:


print("Report: ", classification_report(y_1test, yhat))



