
# coding: utf-8

# In[1]:


get_ipython().system('conda install -y keras')


# In[2]:



get_ipython().system('pip install textgenrnn')


# In[3]:


from textgenrnn import textgenrnn


# In[4]:


textgen = textgenrnn()


# In[5]:


textgen.generate()


# In[6]:


textgen.train_on_texts(open("letranger_clean.txt").readlines(), num_epochs=20)


# In[7]:


textgen.generate()


# In[10]:


poem = textgen.generate(6, temperature=0.9, return_as_list=True)
for line in poem:
    print(line.strip())

