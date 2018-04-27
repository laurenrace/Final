
# coding: utf-8

# In[1]:


lines = [item.strip() for item in open('letranger.txt').readlines()]


# In[2]:


no_camus_lines = [item for item in lines if not(item.startswith("Albert Camus"))]


# In[3]:


import re


# In[4]:


no_weird_brackets_lines = [re.sub(r"\[\d+\]", "", item) for item in no_camus_lines]


# In[5]:


no_weird_brackets_lines


# In[6]:


with open('letranger_clean.txt', 'w') as fh:
    for item in no_weird_brackets_lines:
        fh.write(item + "\n")


# In[7]:


with open('letranger.txt', 'r') as f:
    src = f.read()

# src = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
src += "$" # to indicate the end of the string
model = {}
for i in range(len(src)-2):
    ngram = tuple(src[i:i+2]) # get a slice of length 2 from current position
    next_item = src[i+2] # next item is current index plus two (i.e., right after the slice)
    if ngram not in model: # check if we've already seen this ngram; if not...
        model[ngram] = [] # value for this key is an empty list
    model[ngram].append(next_item) # append this next item to the list for this ngram


# In[8]:


model


# In[9]:


import random
def gen_from_model(n, model, start=None, max_gen=100):
    if start is None:
        start = random.choice(list(model.keys()))
    output = list(start)
    for i in range(max_gen):
        start = tuple(output[-n:])
        next_item = random.choice(model[start])
        if next_item is None:
            break
        else:
            output.append(next_item)
    return output


# In[10]:


str = ''.join(gen_from_model(2, model, ('s', 'p'), 400))
print(str)


# In[11]:


text = open("letranger.txt").read()
words = text.split()


# In[12]:


words.append("$")
model2 = {}
for i in range(len(words)-2):
    ngram = tuple(words[i:i+2]) # get a slice of length 2 from current position
    next_item = words[i+2] # next item is current index plus two (i.e., right after the slice)
    if ngram not in model2: # check if we've already seen this ngram; if not...
        model2[ngram] = [] # value for this key is an empty list
    model2[ngram].append(next_item) # append this next item to the list for this ngram


# In[13]:


model2


# In[14]:


pairs = [(words[i], words[i+1]) for i in range(len(words)-1)]


# In[15]:


pairs = []
for i in range(len(words)-1):
    this_pair = (words[i], words[i+1])
    pairs.append(this_pair)


# In[16]:


pairs[:10]from collections import Counter


# In[17]:


from collections import Counter


# In[18]:


pair_counts = Counter(pairs)


# In[19]:


pair_counts.most_common(20)


# In[20]:


' '.join(gen_from_model(2, model2, ('je', 'ne'), 100))


# In[21]:


get_ipython().system('conda install -y keras')


# In[22]:


get_ipython().system('pip install textgenrnn')


# In[23]:


from textgenrnn import textgenrnn


# In[24]:


textgen = textgenrnn('letranger_weights.hdf5')


# In[ ]:



textgen.generate()


# In[ ]:


textgen.train_on_texts(open("letranger.txt").readlines(), num_epochs=20)


# In[ ]:


poem = textgen.generate(8, temperature=0.4, return_as_list=True)
for line in poem:
    print(line.strip())


# In[ ]:


textgen.save('letranger_weights.hdf5')

