import pandas as pd # for importing csv data into dataframes
import numpy as np # for most commonly array operations
import tensorflow as tf # machine learning library
import seaborn as sb # used for plotting heat maps from confusion matrices
import math
import time
import matplotlib.pyplot as plt # used for plotting figures/graphs
from sklearn.model_selection import train_test_split # used for splitting the data set into train and test set

tf.compat.v1.disable_eager_execution() # enabling the calling of functions from tensorflow v1

tf1 = tf.compat.v1 # abbreviation for TensorFlow v1 compatibility function library

# Function that will create the placeholders of X and Y for future data passing when running a session
def create_placeholders(n_x, n_y):
    
    X = tf1.placeholder(tf.float32, [n_x, None])
    Y = tf1.placeholder(tf.float32, [n_y, None])

    return X, Y

# Function that will convert the labels column/row vector into a matrix with a 1 in the corresponding class row.
def one_hot_matrix(labels, C):

	# creating a tf.constant equal to C (number of classes, depth of matrix)
	C = tf.constant(C, name = 'C')

	one_hot_matrix = tf.one_hot(labels, C, axis = 0)

	sess = tf1.Session()

	one_hot = sess.run(one_hot_matrix)

	sess.close()

	return one_hot

# Function that will initialize the parameters depending on the layers dimensions set for the neural network
def initialize_parameters(layers_dims):
    
    """
        Initializes parameters to build a neural network with Tensor Flow.
        The shapes of the parameters matrices are the following:
        W1 : [64, 17]
        b1 : [64, 1]
        W2 : [32, 64]
        b2 : [32, 1]
        W3 : [32, 32]
        b3 : [32, 1]
        W4 : [16, 32]
        b4 : [16, 1]
        W5 : [4, 16]
        b5 : [4, 1]
    """
    
    L = len(layers_dims)
    parameters = {} # dictionary where parameters will be saved

    """
        Initialization methods
    """
    xavier_initializer = tf.initializers.GlorotUniform()
    #he_initializer = tf.keras.initializers.he_normal(seed=None)
    
    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(xavier_initializer(shape=[layers_dims[l], layers_dims[l-1]]), dtype = tf.float32)
        parameters['b' + str(l)] = tf.Variable(tf.zeros(shape=[layers_dims[l],1]), dtype = tf.float32)
    
    with tf1.Session() as sess:
        init = tf1.global_variables_initializer()
        sess.run(init)
    
    return parameters

# Function that will make the forward propagation of the neural network
def forward_propagation(X, parameters, dropout_prob):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    
    #dropped_A1 = tf.nn.dropout(A1, rate = dropout_prob) # dropout 1
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    
    dropped_A2 = tf.nn.dropout(A2, rate = dropout_prob) # dropout 2

    Z3 = tf.add(tf.matmul(W3, dropped_A2), b3)
    A3 = tf.nn.relu(Z3)
    
    dropped_A3 = tf.nn.dropout(A3, rate = dropout_prob) # dropout 3
    
    Z4 = tf.add(tf.matmul(W4, dropped_A3), b4)
    A4 = tf.nn.relu(Z4)
    
    #dropped_A4 = tf.nn.dropout(A4, rate = dropout_prob) # dropout 4
    Z5 = tf.add(tf.matmul(W5, A4), b5)
    

    return Z5

# Function that will compute the cost
def compute_cost(ZL, Y):

    logits = tf.transpose(ZL) # predictions for that iteration
    labels = tf.transpose(Y) # labels for that iteration
    
    print("cost shapes:", logits.shape, labels.shape)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

# Function that will generate random mini batches from a full data set of X and Y in order to make for gradient descents.
def random_mini_batches(X, Y, mini_batch_size):
    
    m = X.shape[1]
    mini_batches = []
    
    
    permutation = list(np.random.permutation(m)) # generating a random list of column positions from the training examples lenght
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, (k*mini_batch_size) : (k*mini_batch_size)+mini_batch_size]
        mini_batch_Y = shuffled_Y[:, (k*mini_batch_size) : (k*mini_batch_size)+mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# Function that will call all functions mentioned before and make the backpropagation updating the parameters
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):

    # number of units for each layer
    layers_dims = [17, 64, 32, 32, 16, 4]
    
    dropout_prob = tf1.placeholder(tf.float32) # Setting a placeholder for the probability of deleting a node in a layer with dropout
    
    # being n_x the number of input features of the neural network,
    # m the number of training examples and n_y the number of output classes
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[1]

    # lists created in order to display plots
    costs = []
    epochs_accuracies_train = []
    epochs_accuracies_test = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(layers_dims)

    ZL = forward_propagation(X, parameters, dropout_prob)

    cost = compute_cost(ZL, Y)

    optimizer = tf1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf1.global_variables_initializer()

    with tf1.Session() as sess:

        sess.run(init)
        tic = time.time()
        for epoch in range(num_epochs):
            
            epoch_cost = 0
            correct_prediction_epoch = 0
            accuracy_epoch = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train.T, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_accuracy = 0
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, dropout_prob: 0.2})
                
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                #print ("Accuracy of epoch %i: %f" % (epoch, epoch_accuracy))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                # calculating accuracies per epoch of training and test set
                correct_prediction_epoch = tf.equal(tf.argmax(ZL), tf.argmax(Y))
                accuracy_epoch = tf.reduce_mean(tf.cast(correct_prediction_epoch, "float"))
                epochs_accuracies_train.append(accuracy_epoch.eval({X: X_train, Y: Y_train.T, dropout_prob: 0}))
                epochs_accuracies_test.append(accuracy_epoch.eval({X: X_test, Y: Y_test.T, dropout_prob: 0}))
            


        parameters = sess.run(parameters)
        print ("Parameters have been trained!!")

        # Getting the accuracies for the hole train and test set
        
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train.T, dropout_prob: 0}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test.T, dropout_prob: 0}))
        
        preds = tf.argmax(ZL)
        labels = tf.argmax(Y)
        confusion_matrix = tf.math.confusion_matrix(preds, labels) # generating the confusion matrix with the predictions and the labels.
        
        # Figure for the cost descent
        f = plt.figure(1)
        plt.plot(np.squeeze(costs), color='red')
        plt.ylabel('cost')
        plt.xlabel('iteration per epoch')
        plt.title("Learning rate= " + str(learning_rate))
        f.show()
        
        # Figure for the accuracies of the train/test set
        g = plt.figure(2)
        plt.plot(np.squeeze(epochs_accuracies_train))
        plt.plot(np.squeeze(epochs_accuracies_test))
        plt.ylabel('accuracy')
        plt.xlabel('accuracy per epoch')
        g.show()
        
        # heat map of the confusion matrix
        i = plt.figure(3)
        heat_map = sb.heatmap(confusion_matrix.eval({X: X_test, Y: Y_test.T, dropout_prob: 0}), cmap="YlGnBu")
        plt.ylabel('Predicted values')
        plt.xlabel('Real values')
        i.show()
        
        plt.show()
        
        #print("preds: ", preds.eval({X: X_test, Y: Y_test.T}))
        #print("labels: ", labels.eval({Y: Y_test.T}))
        print("confusion_matrix: \n", confusion_matrix.eval({X: X_test, Y: Y_test.T, dropout_prob: 0}))
        
    
        toc = time.time()
        print("Time spent: ", (toc-tic)/60)
    
    return parameters



# main
# getting the path of the csv file
filename = '/Users/rgomezr/Documents/OneDrive/DOCUMENTS/TFG/dev/MLP/computational_data/Jamones-6etapas(Ccomputacionales)_1_without_class3_and_4.csv'
# getting all data in the csv in a panda data frame matrix
ham_data = pd.read_csv(filename, float_precision = 'round_trip')

print(ham_data)
print("ham data shape", ham_data.shape)

#print(ham_data.head())

cols = [col for col in ham_data.columns if col not in ['NOMBRE (jamón+día+músculo+imagen+ROI)', 'healing_period']]

data = ham_data[cols]
#print(data)

target = ham_data['healing_period']
#print(target)

#todo: change number of classes again
labels = one_hot_matrix(target, 4).T

#print("data",data.T)

#print(labels)

print(data.shape, labels.shape)

# splitting all data and labels into training and testing data in order to use with the model
train_X_orig, test_X, train_Y_orig, test_Y = train_test_split(data, labels, test_size = 0.1, random_state = 10)

train_X = train_X_orig.T # Transpose train_X_orig, to have shape (Nx,m)
train_X = np.float32(train_X) # cast X from float64 to float32
train_Y = train_Y_orig

test_X = test_X.T
test_X = np.float32(test_X)

print(train_X.shape, test_X.shape)

# obtaining the trained parameters once the model
# has trained with the training data
parameters = model(train_X, train_Y, test_X, test_Y)
