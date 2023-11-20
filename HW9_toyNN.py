import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# read_data, get_df_shape, data_split are the same as HW3
def read_data(filename: str) -> pd.DataFrame:
    d = pd.read_csv(filename)
    df = pd.DataFrame(data=d)
    return df

def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Filter the dataframe to include only Setosa and Virginica rows
    filtered_df = df[df['variety'].isin(['Setosa', 'Virginica'])]
    # Extract the required features and labels from the filtered dataframe
    #features = filtered_df[['sepal.length', 'sepal.width']]
    #features = filtered_df[['sepal.length', 'petal.width']]
    #features = filtered_df[['petal.length', 'sepal.width']]
    features = filtered_df[['petal.length', 'petal.width']]
    labels = filtered_df['variety'].map({'Setosa': 0, 'Virginica': 1})
    return features, labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def function_derivative(z):
    fd = sigmoid(z)
    return fd * (1 - fd)


# At the start of the class, initialize the weights and biases
# Take an intial guess at the weights and biases at epoch zero
# Then when we do the forward pass, we will get an output
# Then we will do the backpropagation pass:
    # We will calculate the error between the output and the target
    # Then we will calculate the derivative of the cost function with respect to the weights and biases
    # Then we will update the weights and biases
    # Then we will repeat the process until we get to the end of the epoch
    # Then we will repeat the process for the next epoch
    # Then we will make predictions on the test data







class NN:
    def __init__(self, features, hidden_neurons, output_neurons, learning_rate):
        self.features = features
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        
        # initialize weights
        # nueron layer of the hidden layer
        self.V = np.random.randn(self.features, self.hidden_neurons)
        # nueron layer of teh output layer
        self.W = np.random.randn(self.hidden_neurons, self.output_neurons)
        
        # initialize biases: 0
        self.V0 = np.zeros((self.hidden_neurons))
        self.W0 = np.zeros((self.output_neurons))
    
    def train(self, X, t, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # forward pass big guess
            
            
            # This is the hidden layer
            net_u = X.dot(self.V) + self.V0
            H = sigmoid(net_u)
            
            
            # This is the output layer
            net_z = H.dot(self.W) + self.W0
            O = sigmoid(net_z)
            
            # backpropagation pass
            error_output = O - t
            # this is the derivative of the cost function
          
          
            # This is output layer updates
          
            # This will take 
            d_W = H.T.dot(error_output * function_derivative(net_z))
            # this is the derivative of the cost function with respect to the bias which will update the bias
            d_W0 = np.sum(error_output * function_derivative(net_z), axis=0)
            # this is the derivative of the cost function with respect to the weights which will update the weights

            # d_w is the derivative of the cost function with respect to the weights
            # d_wo is the derivative of the cost function with respect to the biases
            
            
            
            #This is hidden layer updates
            
            #in a nueral network you update both the weights and the baises during each pass of the backpropagation
            error_hidden_layer = error_output.dot(self.W.T) * function_derivative(net_u)
            # this is the derivative of the cost function with respect to the weights which will update the weights
            d_V = X.T.dot(error_hidden_layer)
            # this is the derivative of the cost function with respect to the bias which will update the bias
            d_V0 = np.sum(error_hidden_layer, axis=0)
            
            
            
            
            
            
            # update weights and biases
            self.W -= self.learning_rate * d_W
            self.W0 -= self.learning_rate * d_W0
            self.V -= self.learning_rate * d_V
            self.V0 -= self.learning_rate * d_V0
            
            #find the cost function
            if epoch % 10 == 0:
                loss =  np.square(np.subtract(t,O)).mean() 
                costs.append(loss)
            if epoch %50 == 0:
                print("Epoch: %d, cost: %.4f" % (epoch, loss))
                
        return costs
    
    def predict(self, X):
        output = X
        for i in range(len(self.weights) - 1):
            output = sigmoid(output.dot(self.weights[i]) + self.biases[i])
        # For the final layer
        final_output = sigmoid(output.dot(self.weights[-1]) + self.biases[-1])
        return (final_output > 0.5).astype(int)
    
if __name__ == "__main__":
    def accuracy(t, y_pred):
        accuracy = np.sum(t == y_pred) / len(t)
        return accuracy

    # Read the data
    train_df = read_data("iris_training_data.csv")
    test_df = read_data("iris_testing_data.csv")
    
    X, t =  extract_features_label(train_df)
    X_test, t_test = extract_features_label(test_df)

    t = t.values.reshape([len(t),1])
    t_test = t_test.values.reshape([len(t_test),1])
    
# create Neural Network class
nn = NN(features=2, hidden_neurons=5, output_neurons=1, learning_rate=0.01)

# train the network
cost = nn.train(X, t)

# make predictions on the test data
y_pred = nn.predict(X_test)

# evaluate the accuracy
acc = accuracy(t_test,y_pred)
print(acc)
  
plt.plot(cost)
plt.show()