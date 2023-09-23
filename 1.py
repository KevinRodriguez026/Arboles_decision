#!/usr/bin/env python
# coding: utf-8

# Arboles de decisión

# In[9]:


import pandas as pd
file_path = r'C:\Users\Kevin Rodriguez\Documents\Arboles_de_decision\wine.data'
data = pd.read_csv(file_path, header=None)
data.head()


# In[15]:


data.shape


# In[16]:


data.describe()


# In[12]:


from sklearn.model_selection import train_test_split
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[14]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


# In[22]:


tree = DecisionTreeClassifier()


# In[23]:


arbol = tree.fit(X_train, y_train)


# In[24]:


plot_tree(arbol)


# In[25]:


predicciones = arbol.predict(X_test)


# In[29]:


y_verdadero = [1, 2, 3, ...] 
precision = accuracy_score(y_test, predicciones)
print("Precisión de las predicciones:", precision)


# In[ ]:




