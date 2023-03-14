#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import csv
from sklearn import preprocessing
from string import punctuation
from sklearn import metrics
import matplotlib.pyplot as plt
import random


# In[2]:


reviews = []
labels = []
test = []
test_ids = []
external_authors = []
external_lines = []
with open('./text.csv', 'r',encoding="latin-1") as f:
    text_reader = csv.reader(f,delimiter=",")
    next(text_reader)
    for row in text_reader:
        reviews.append(row[0])
with open('./author.csv', 'r') as labels_csv:
    author_reader = csv.reader(labels_csv,delimiter =",")
    next(author_reader) #ignore header
    for row in author_reader:
        labels.append(row[0])
with open("./test.csv",'r') as test_csv:
    text_reader = csv.reader(test_csv, delimiter= ",")
    next(text_reader) #ignore header
    
    for row in text_reader:
        test_ids.append(row[0])
        test.append(row[1])
        
with open("./newData.csv",'r') as external_data:
    text_reader = csv.reader(external_data,delimiter=",")
    
    for row in text_reader:
        line = row[1].strip()
        
        if len(line.split() )> 5: #if the line has at least 5 words, keep it
            external_authors.append(row[0].strip())
            external_lines.append(line)


# In[3]:


len(reviews)


# In[4]:


len(test)


# In[5]:


len(external_authors)


# In[6]:


reviews[:20]


# In[7]:


labels[:20]


# In[8]:


test[:20]


# In[9]:


external_lines[:20]


# In[10]:


from string import punctuation
#all_text = ' '.join([c for c in reviews + test if c not in punctuation])
all_text = ' '.join([c for c in reviews + test +external_lines])

words = all_text.split()


# In[11]:


all_text[:305]


# In[12]:


words[:100]


# In[13]:


reviews[2]


# In[14]:


# Create your dictionary that maps vocab words to integers here
vocab_to_int = {word:index for index,word in enumerate(set(words),1)}
vocab_to_int["<PAD>"] = 0

# Convert the reviews to integers, same shape as reviews list, but with integers
reviews_ints = []
for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])
    
test_ints = []
for test_line in test:
    test_ints.append([vocab_to_int[word] for word in test_line.split()])
    
external_ints = []

for external_line in external_lines:
    external_ints.append([vocab_to_int[word] for word in external_line.split()])


# In[15]:


len(vocab_to_int)


# In[16]:


reviews_ints[1]


# In[17]:


test_ints[1]


# In[18]:


labels_to_int = {}
int_to_labels = {}
unique_labels = list(set(labels))
for i,label in enumerate(unique_labels):
    labels_to_int[label] = i
    int_to_labels[i] = label
    
int_labels = []

for label in labels:
    int_labels.append(labels_to_int[label])
    
int_external_labels = []

for label in external_authors:
    int_external_labels.append(labels_to_int[label])
    
print(labels_to_int)
print(int_to_labels)
print(int_labels[:10])


# In[19]:


encoder = preprocessing.LabelBinarizer()
encoder.fit(list(set(int_labels)))
one_hot_labels = encoder.transform(int_labels)
                                   
one_hot_labels

one_hot_external_labels = encoder.transform(int_external_labels)

one_hot_external_labels


# In[20]:


from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
print("Minimum length: {}".format(min(review_lens)))
print("Average length: {}".format(sum(review_lens)/len(review_lens)))


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
text_lens_list = list(review_lens)
p,h,s = plt.hist(text_lens_list,bins = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900])


# In[22]:


for  i in range(len(p)):
    print(p[i]/len(text_lens_list)  ,"% is between ",h[i]," and ",h[i+1])


# In[23]:


test_lens = Counter([len(x) for x in test_ints])
print("Zero-length reviews: {}".format(test_lens[0]))
print("Maximum review length: {}".format(max(test_lens)))
print("Minimum length: {}".format(min(test_lens)))
print("Average length: {}".format(sum(test_lens)/len(test_lens)))


# In[24]:


seq_len = 100
features = []

for review in reviews_ints:
    review_size = len(review)
    if review_size < seq_len:
        padded_review = [0] * seq_len
        padded_review[seq_len-len(review):seq_len] = review
    elif review_size > seq_len:
        padded_review = review[:seq_len]
    
    features.append(padded_review)
features  = np.array(features)


# In[25]:


features[:10,:100]


# In[26]:


test_features = []

for test_line in test_ints:
    line_size = len(test_line)
    if line_size < seq_len:
        padded_line = [0] * seq_len
        padded_line[seq_len-len(test_line):seq_len] = test_line
    elif line_size > seq_len:
        padded_line = test_line[:seq_len]
        
    test_features.append(padded_line)

test_features = np.array(test_features)


# In[27]:


test_features.shape


# In[28]:


test_features[:10,:100]


# In[29]:


external_features = []

for external_line in external_ints:
    line_size = len(external_line)
    if line_size < seq_len:
        padded_line = [0] * seq_len
        padded_line[seq_len-len(external_line):seq_len] = external_line
    elif line_size > seq_len:
        padded_line = external_line[:seq_len]
        
    external_features.append(padded_line)

external_features = np.array(external_features)


# In[30]:


split_frac = 0.80

split_index  = int(len(features)*split_frac)

train_x, val_x = features[:split_index],features[split_index:]
train_y, val_y = one_hot_labels[:split_index],one_hot_labels[split_index:]

split_index = int(len(val_x)/2)

val_x, test_x = val_x[:split_index],val_x[split_index:]
val_y, test_y = val_y[:split_index],val_y[split_index:]

# add external data to traininig set
train_x = np.append(train_x, external_features,axis = 0)
train_y = np.append(train_y, one_hot_external_labels,axis= 0)
random_indexes = np.random.permutation(train_x.shape[0])

train_x = train_x[random_indexes]
train_y = train_y[random_indexes]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), train_y.shape, 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# In[31]:


lstm_size = 512
fully_connected_size = 20
lstm_layers = 1
fully_connected_layers = 2
batch_size = 512
learning_rate = 0.005
dropout_lstm = 1
dropout_fully_connected = 1
display_every_iterations = 20 


# In[32]:





# In[32]:


def get_configuration_string(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout):
    configuration_string = "lstm_layers={}&lstm_size={}&fully_connected_layers={}&fully_connected_size={}&batch_size={}&learning_rate={}&lstm_dropout={}&fully_connected_dropout={}"     .format(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout)
    
    return configuration_string
    
get_configuration_string(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,dropout_lstm,dropout_fully_connected)


# In[33]:


n_words = len(vocab_to_int)

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32,shape=[batch_size,None],name="inputs")
    labels_ = tf.placeholder(tf.int32,shape=[batch_size,len(unique_labels)],name = "labels")
    keep_prob = tf.placeholder(tf.float32,name ="keep_prob")
    fully_connected_keep_prob = tf.placeholder(tf.float32,name = "fc_keep_prob")
    learning_rate_ = tf.placeholder(tf.float32,name = "learning_rate") 


# In[34]:


# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words,embed_size),-0.5,0.5))
    embed = tf.nn.embedding_lookup(embedding,inputs_)


# In[35]:


with graph.as_default():
    # Your basic LSTM cell
    #lstm = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
    
    # Add dropout to the cell
    #drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    #cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
    cell_list = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size) ,output_keep_prob=keep_prob)  ]
    cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# In[36]:


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell,embed,initial_state=initial_state)


# In[37]:


with graph.as_default():
    fully_connected = tf.contrib.layers.fully_connected(outputs[:, -1], fully_connected_size, activation_fn=tf.nn.relu)
    fully_connected = tf.nn.dropout(fully_connected,fully_connected_keep_prob)
    
    
    for layer in range(fully_connected_layers - 1):
        fully_connected = tf.contrib.layers.fully_connected(fully_connected, fully_connected_size, activation_fn=tf.nn.relu)
        fully_connected = tf.nn.dropout(fully_connected,fully_connected_keep_prob)
        
    fully_connected = tf.contrib.layers.fully_connected(fully_connected, 3, activation_fn=tf.nn.relu)
    logits = tf.identity(fully_connected)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=labels_))
    
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)


# In[38]:


with graph.as_default():
    predictions = tf.nn.softmax(logits)
    predictions_hardmax = tf.argmax(predictions,1)
#    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[39]:


def get_batches(x, y, batch_size=100):
    #shuffle batches at every ecoch    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[40]:


def calc_classification_metrics(predictions,real_values):
    accuracy =  sum(predictions == real_values)/predictions.shape[0] # metrics.accuracy_score(predictions,real_values)
    error = 1 - accuracy
    precision = 0# metrics.precision_score(predictions,real_values)
    recall = 0#metrics.recall_score(predictions,real_values)
    
    return accuracy,error,precision,recall


# In[41]:


# Function to generate a table for manual error analysis
# compares prediction vs real values, and if different generates a row in analysis table
# the columns of the table represent the possible errors in the form correct_label:<author>_predicted_label:<author>
# these columns store 1 for the corresponding error in the line, 0 for the rest
def error_analysis(configuration_string,predictions,real_values):
    headers = []
    
    for correct_label in int_to_labels.values():
        for predicted_label in int_to_labels.values():
            if correct_label != predicted_label:
                analysis_column = "correct_label:"+correct_label+"_predicted_label:"+predicted_label
                headers.append(analysis_column)
                    
    with open("error_analysis/"+configuration_string+".csv","w") as error_file:
                header_writer = csv.writer(error_file)
                writer = csv.DictWriter(error_file, fieldnames= headers, quoting=csv.QUOTE_ALL)
                header_writer.writerow(headers)
    
                for i in range(predictions.shape[0]):
                    analysis_colums = {}

                    for header in headers:
                        analysis_colums[header] = 0

                    if predictions[i] != real_values[i]:
                        correct_label = int_to_labels[real_values[i]]
                        predicted_label = int_to_labels[predictions[i]]
                        analysis_column = "correct_label:"+correct_label+"_predicted_label:"+predicted_label
                        analysis_colums[analysis_column]  = 1
                        writer.writerow(analysis_colums)
            
            
#error_analysis("test",predictions,real_values)


# In[42]:


def train(graph,saver,epochs,lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout):
    
    log_file = open("log_book.csv","a")
    csv_writer = csv.writer(log_file,delimiter = ",")
    
    configuration_string = get_configuration_string(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout)
    print("Starting training for: ",configuration_string)
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range(epochs):
            state = sess.run(initial_state)
        
            
            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed_dict = {inputs_: x,
                    labels_: y,
                    keep_prob: lstm_dropout,
                    fully_connected_keep_prob:fully_connected_dropout,
                    initial_state: state,
                        learning_rate_ : learning_rate}
                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed_dict)
            
                if iteration%display_every_iterations ==0:
                    val_acc = []
                    val_costs = []
                
                    train_prediction_hardmax = sess.run(predictions_hardmax,feed_dict=feed_dict)
                    train_real_hardmax = np.argmax(y,1)
                    train_accuracy,train_error,train_precision,train_recall  = calc_classification_metrics(train_prediction_hardmax,train_real_hardmax)
                
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                
                    epoch_predictions = np.array([])
                    epoch_real_values = np.array([])
                    for x, y in get_batches(val_x, val_y, batch_size):
                        feed_dict = {inputs_: x,
                            labels_: y,
                            keep_prob: 1,
                            fully_connected_keep_prob:1,
                            initial_state: val_state,
                            learning_rate_ : learning_rate}
                    
                        val_prediction = sess.run(predictions,feed_dict=feed_dict)
                        val_cost = sess.run(cost,feed_dict=feed_dict)
                        val_prediction_hardmax = sess.run(predictions_hardmax,feed_dict=feed_dict)
                        val_real_hardmax = np.argmax(y,1)
                        val_accuracy,val_error,val_precision,val_recall  = calc_classification_metrics(val_prediction_hardmax,val_real_hardmax)
                        val_acc.append(val_accuracy)
                        val_costs.append(val_cost)
                        epoch_predictions= np.append(epoch_predictions,val_prediction_hardmax,axis = 0)
                        epoch_real_values = np.append(epoch_real_values,val_real_hardmax ,axis = 0)
                        
                    val_cost = np.mean(val_costs)  
                    val_acc  = np.mean(val_acc)
                    print("Epoch: {}/{}".format(e+1, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss),
                      "Train accuracy: {:.3f}".format(train_accuracy),
                     "Train error: {:.3f}".format(train_error),
                      "Val cost: {:.3f}".format(val_cost),
                      "Val acc: {:.3f}".format(val_acc)
                     )
                    
                

                    
                iteration +=1
            learning_rate = 0.90 * learning_rate
        saver.save(sess, "checkpoints/{}/checkpoint.ckpt".format(configuration_string))
        
        csv_writer.writerow([configuration_string,loss,train_accuracy,train_error,val_cost,val_acc,(1-val_acc)])
        log_file.close()
        
        epoch_predictions = np.array(epoch_predictions)
        epoch_real_values = np.array(epoch_real_values)
        
        error_analysis(configuration_string,epoch_predictions,epoch_real_values)
        
        print("Finished training for: ",configuration_string)


# In[43]:


# original run with original parameters for 10 epochs

dropout_lstm = 1
dropout_fully_connected = 1

with graph.as_default():
    saver = tf.train.Saver()
    
train(graph,saver,2,lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,dropout_lstm,dropout_fully_connected)


# In[44]:


with graph.as_default():
    saver = tf.train.Saver()

# THE FOLLOWING LINES WERE USED TO WRITE THE HEADER TO THE LOG BOOK, ARE NECCESARY ONLY IF LOG BOOK ITS EMPTIED    
#log_file = open("log_book.csv","a")
#csv_writer = csv.writer(log_file,delimiter = ",")
#header = ["config_string","train_loss","train_accuracy","train_error","validation_loss","validation_accuracy","validation_error"]
#csv_writer.writerow(header)
#log_file.close()


for i in range(15):
    #lstm_dropout = random.random()
    #fully_connected_dropout = random.random()
    
    #train(graph,saver,20,lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout)
    print("")
    print("--------------------------------------------------------------------------------------------")


# In[45]:


test_acc = []
test_costs = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/lstm_layers=1&lstm_size=512&fully_connected_layers=2&fully_connected_size=20&batch_size=512&learning_rate=0.005&lstm_dropout=1&fully_connected_dropout=1'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        print(x.shape,y.shape)
        feed_dict = {inputs_: x,
                labels_: y,
                keep_prob: 1,
                fully_connected_keep_prob:1,
                initial_state: test_state,
                        learning_rate_ : learning_rate}
        
        test_prediction = sess.run(predictions,feed_dict=feed_dict)
        test_cost = sess.run(cost,feed_dict=feed_dict)
        test_prediction_hardmax = sess.run(predictions_hardmax,feed_dict=feed_dict)
        test_real_hardmax = np.argmax(y,1)
        test_accuracy,test_error,test_precision,test_recall  = calc_classification_metrics(test_prediction_hardmax,test_real_hardmax)
        test_costs.append(test_cost)
        test_acc.append(test_accuracy)
        print("test cost",test_cost , test_accuracy )
        #batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        #test_acc.append(batch_acc)
        #print(x.shape,y.shape)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc) ),
         "Test cost: {:.3f}".format(np.mean(test_costs)))


# In[46]:


with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/lstm_layers=1&lstm_size=512&fully_connected_layers=2&fully_connected_size=20&batch_size=512&learning_rate=0.005&lstm_dropout=1&fully_connected_dropout=1'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    
    full_batches =  int(test_features.shape[0]/batch_size)
    submit_file = open("submission.csv","w")
    csv_writer = csv.writer(submit_file,delimiter = ",")
    header = ["id","EAP","HPL","MWS"]
    csv_writer.writerow(header)
    
    # TODO: find a way to send 1 line at a time without having to take special care of last incomplete batch
    for i in range(full_batches):
        batch = test_features[ (i*batch_size):((i+1)*batch_size)]

        feed_dict = {inputs_: batch,
                keep_prob: 1,
                fully_connected_keep_prob:1,
                initial_state: test_state,
                        learning_rate_ : learning_rate}
    
        test_prediction = sess.run(predictions,feed_dict=feed_dict)
        test_prediction_hardmax = np.argmax(test_prediction,1)
    
        for j,prediction in enumerate(test_prediction_hardmax):
            EAP_prob = test_prediction[j,labels_to_int["EAP"]]
            HPL_prob = test_prediction[j,labels_to_int["HPL"]]
            MWS_prob = test_prediction[j,labels_to_int["MWS"]]
        
            line = [test_ids[(i*batch_size + j)] , EAP_prob, HPL_prob, MWS_prob]
            csv_writer.writerow(line)
        
        print("finished batch {}".format(i))
            #print("Finished writing {}".format(i))
            #print(test_ids[i] ,test[i][:50] , " ", int_to_labels[test_prediction_hardmax] , EAP_prob, HPL_prob, MWS_prob)
       
    if test_features.shape[0]%batch_size != 0:
        print("Last minibatch" ,full_batches*batch_size,",", test_features.shape[0])
        
        i+=1
        batch = np.zeros((batch_size,test_features.shape[1]))
        batch[0:(test_features.shape[0]-full_batches*batch_size),:] = test_features[full_batches*batch_size:,:]
        
        feed_dict = {inputs_: batch,
                keep_prob: 1,
                fully_connected_keep_prob:1,
                initial_state: test_state,
                        learning_rate_ : learning_rate}
    
        test_prediction = sess.run(predictions,feed_dict=feed_dict)
        test_prediction_hardmax = np.argmax(test_prediction,1)

        for j in range(test_features.shape[0]-full_batches*batch_size):
            EAP_prob = test_prediction[j,labels_to_int["EAP"]]
            HPL_prob = test_prediction[j,labels_to_int["HPL"]]
            MWS_prob = test_prediction[j,labels_to_int["MWS"]]
        
            line = [test_ids[(i*batch_size + j)] , EAP_prob, HPL_prob, MWS_prob]
            csv_writer.writerow(line)
        
        print("finished batch {}".format(i))

    submit_file.close()


# In[47]:




