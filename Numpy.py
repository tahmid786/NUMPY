#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
i=np.array([1,2,3])
b=np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(i)


# In[3]:


#dimension 
print(i.ndim)


# In[4]:


#shape(row,coloumn)
print(i.shape,b.shape)


# In[13]:


#get data type -dtype
i=np.array([1,2,3])
print(i.dtype,b.dtype)
c=np.array([1,2,3],dtype='int16')
print(c.dtype)


# In[14]:


#get total item-itemsize
print(i.size,c.size,b.size)


# In[15]:


#get total size-nbytes
print(i.nbytes,c.nbytes,b.nbytes)


# In[16]:


#change/access elements,rows,coloumn
a=np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
a.shape


# In[17]:


#get a specific element-[r,c]
print(a[1,5])
print(a[1,-2])


# In[18]:


#get specific row
print(a[0,:])
#get specific column
print(a[:,3])


# In[19]:


#getting little more fancy[startIndex:endIndex:stepSize] endindex+1 
print(a[0,1:7])#end-index 6 but 6+1=7


# In[20]:


print(a[0,1:6:2])


# In[21]:


#change element 3
a[0,2]=9
print(a)


# In[22]:


#all 1st elements 5
a[0,:]=5
print(a[0,:])
a[0,:]=[1,2,3,4,5,6,7]
print(a[0,:])


# In[23]:


#3d example 
#paste it and ask chatgpt for difference
b=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])#[1,2]->2,[[1,2],[3,4]]->2,[[[1,2],[3,4]],[[5,6],[7,8]]]->2  (2,2,2)
c=np.array([[[1,2],[3,4],[5,6],[7,8]]])#1d=[1,2]->2, 2d=[[1,2],[3,4],[5,6],[7,8]]->4, 3d=[[[1,2],[3,4],[5,6],[7,8]]]->1  (1,4,2)
print(b.ndim,c.ndim)
print(b.shape,c.shape)


# In[24]:


#b->4
print(b[0,1,1])


# In[25]:


#b->[[1,2],[3,4]]
print(b[0,:,:])


# In[26]:


#b->[[3,4],[7,8]]
print(b[:,1,:])


# In[27]:


#b->[[3,4],[7,8]] change value
b[:,1,:]=[[9,9],[8,8]]
print(b)


# In[28]:


#all 0 matrix np.zeros(shape)
#all 1 matrix np.ones(shape)
print(np.zeros(5))
print(np.zeros((2,3)))
print(np.zeros((3,2,1)))


# In[3]:


#all matrix with any value np.full(shape,value)
np.full((4,1,1),3)


# In[6]:


#passing array a,i np.fuLL_like ()
a=np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
np.full_like(a,3)


# In[4]:


#random decimal numbers np.random.rand(4,2,3)
np.random.rand(4,2,3)


# In[29]:


#random sample np.random.random_sample(random_array.shape)
np.random.random_sample(a.shape)


# In[2]:


#random integer value np.random.randint(low, high=None, size=None, dtype=int)
import numpy as np
np.random.randint(-3, high=3, size=(3,3), dtype=int)


# In[10]:


#identity matrix
np.identity(4)


# In[36]:


#repeating values np.repeat(array, repeats, axis=None)
arr=np.array([1,2,3])
print(np.repeat(arr,3,axis=0))
arr=np.array([[1,2,3]])
#for 2d
print(np.repeat(arr,3,axis=0))#0->y axis
print(np.repeat(arr,3,axis=1))#1->x axis


# In[62]:


#solve
r=np.full((5,5),1)
print(r)
r[1:4, 1:4]=0
print(r)
r[2,2]=9
print(r)



# In[3]:


#careful of copying numpy
import numpy as np
a=np.array([1,2,3])
b=a
b[0]=100
print(b)
print(a)


# In[4]:


a=np.array([1,2,3])
b=a.copy()
b[0]=100
print(b)
print(a)


# In[9]:


#mathematics
#elementwise operation
a=np.array([1,2,3,4])
print(a*2)
a+=3
print(a)
a**2
print(a)


# In[6]:


#take sine
import numpy as np
a=np.array([1,2,3,4])
np.sin(a)


# In[7]:


np.cos(a)
#https://docs.scipy.org


# In[9]:


#matrix multiplication
a=np.full((2,3),2)
b=np.full((3,2),3)
c=np.matmul(a,b)
print(c)


# In[12]:


#determinant
a=np.identity(3)
b=np.linalg.det(a)
print(b)


# In[13]:


#statistics
stats=np.array([[1,2,3],[4,5,6]])
print(np.min(stats))
print(np.max(stats))
print(np.max(stats,axis=1))


# In[14]:


a=np.sum(stats)
print(a)


# In[3]:


#reorganizing array
import numpy as np
before=np.array([[1,2,3,4],[5,6,7,8]])
print(before.shape)
after=before.reshape((8,1))
print(after)
after=before.reshape((2,2,2))
print(after)


# In[11]:


#vertically stacking vectors
import numpy as np
v1=np.array([1,2,3,4])
v2=np.array([3,4,5,6])
v=np.vstack([v1,v2,v2,v1])
print(v)
h1=np.zeros((3,2))
h2=np.ones((3,3))
h=np.hstack((h1,h2))
print(h)


# In[37]:


#load data from file
import numpy as np
filedata=np.genfromtxt('C:\\Users\\tahmi\\OneDrive\\Desktop\\data.txt', delimiter=',',dtype=int)
print(filedata)
x=filedata.astype('float64')
print(x)


# #### boolean masking and advanced indexing
# 

# In[35]:


print(filedata>300)
print(filedata[filedata>300])


# In[36]:


a=np.any(filedata>100,axis=0)
print(a)


# In[38]:


((filedata>200)&(filedata<900))


# In[40]:


~((filedata>200)&(filedata<900))


# In[ ]:





# In[ ]:





# In[ ]:




