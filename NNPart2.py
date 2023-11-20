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

def extract_features_label(df: pd.DataFrame, filter, filter2, numfeatures, feat1 = "", feat2 = "", feat3 = "", feat4 = "") -> Tuple[pd.DataFrame, pd.Series]:
    # Filter the dataframe to include only Setosa and Virginica rows
    filtered_df = df[df['variety'].isin([filter, filter2])]
    # Extract the required features and labels from the filtered dataframe
    #features = filtered_df[['sepal.length', 'sepal.width']]
    #features = filtered_df[['sepal.length', 'petal.width']]
    #features = filtered_df[['petal.length', 'sepal.width']]
    if numfeatures == 1:
        features = filtered_df[[feat1]]
    elif numfeatures == 2:
        features = filtered_df[[feat1, feat2]]
    elif numfeatures == 3:
        features = filtered_df[[feat1, feat2, feat3]]
    elif numfeatures == 4:
        features = filtered_df[[feat1, feat2, feat3, feat4]]
    labels = filtered_df['variety'].map({filter: 0, filter2: 1})
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

class NNetwork:
    def __init__(self, features, hidden_layers, output_neurons, learning_rate):
        self.features = features
        self.hidden_neurons = hidden_layers
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        
        # initialize weights
        num_of_layers = len(hidden_layers)
        print(num_of_layers)
        for i in range(num_of_layers):
            if i == 0:
                self.weights.append(np.random.randn(self.features, self.hidden_neurons[i]))
                self.biases.append(np.zeros((self.hidden_neurons[i])))
            else:
                self.weights.append(np.random.randn(self.hidden_neurons[i-1], self.hidden_neurons[i]))
                self.biases.append(np.zeros((self.hidden_neurons[i])))
         
        self.weights.append(np.random.randn(self.hidden_neurons[-1], self.output_neurons))
        self.biases.append(np.zeros((self.output_neurons)))    
    #gets to comp.licated in teh same train function
    def forward_pass(self, X):
        passes = []
        layers = []
        for i in range(len(self.weights)):
            if i == 0:
                net_u = X.dot(self.weights[i]) + self.biases[i]
                H = sigmoid(net_u)
                layers.append(net_u)
                passes.append(H)
            else:
                net_u = passes[i-1].dot(self.weights[i]) + self.biases[i]
                H = sigmoid(net_u)
                layers.append(net_u)
                passes.append(H)
        return passes, layers

    
    def backward_pass(self, X, t, forward_pass, layers):
        # Error at the output layer
        error = forward_pass[len(forward_pass)-1] - t
        update_weights = []
        update_biases = []
        #weights = V,U,W
        #passes = H1,H2,O
        #laters = net_u, net_v,net_z
        for i in range(len(self.weights)):  # Start from second-last layer
            if i == len(self.weights) - 1:
                grad_w = X.T.dot(error)
                grad_b = np.sum(error, axis=0)
            else:
                grad_w = forward_pass[len(forward_pass)-i-2].T.dot(error * function_derivative(layers[len(layers)-i-1]))
                grad_b = np.sum(error * function_derivative(layers[len(layers)-i-1]), axis=0)
                error = error.dot(self.weights[len(self.weights)-i-1].T) * function_derivative(layers[len(layers)-i-2])
            update_weights.append(grad_w)
            update_biases.append(grad_b)

        return update_weights[::-1], update_biases[::-1]
        
    def train(self, X, t, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # Forward pass
            Fp,layers = self.forward_pass(X)
            
            Uw, Ub = self.backward_pass(X, t, Fp,layers)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * Uw[i]
                self.biases[i] -= self.learning_rate * Ub[i]
            loss = np.square(np.subtract(t, Fp[-1])).mean()  # Use the last element of Fp for cost
            #find the cost function
            if epoch % 10 == 0:
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
    def get_feature(n):
        if n==1:
            return "sepal.length"
        if n==2:
            return "sepal.width"
        if n==3:
            return "petal.length"
        if n==4:
            return "petal.width"
        if n>4 or n<1:
            print("Invalid feature number")
            return None
    
    train_df = read_data("iris_training_data.csv")
    test_df = read_data("iris_testing_data.csv") 
    
    print("Welcome to the Neural Network: ")
    num_features = int(input("How many features do you want to use? (1-4) "))
    features = []
    
    for i in range(num_features):
        feature = int(input("What feature do you want to use? type 1 for sepal length, 2 for sepal width, 3 for petal length, 4 for petal width "))
        if feature == None:
            print("Invalid feature")
            break
        else:
            features.append(feature)
            
    hidden_layers = int(input("How many hidden layers do you want? "))
    hidden_layers_arr = []
    for i in range(hidden_layers):
        neurons = int(input(f"How many neurons do you want in layer{i+1}? "))
        hidden_layers_arr.append(neurons)
    output_neurons = int(input("How many output neurons do you want? "))
    learning_rate = float(input("What learning rate do you want? "))
    comp1 = ""
    comp2 = ""
    compare = int(input("What do you want to compare? type 1 for setosa and versicolor, 2 for setosa and virginica, 3 for versicolor and virginica "))
    if compare == 1:
        comp1 = "Setosa"
        comp2 = "Versicolor"
    elif compare == 2:
        comp1 = "Setosa"
        comp2 = "Virginica"
    elif compare == 3:  
        comp1 = "Versicolor"
        comp2 = "Virginica"
    
    if num_features == 1:
        X,t = extract_features_label(train_df,comp1,comp2, num_features, get_feature(features[0]))
        x_test, t_test = extract_features_label(test_df,comp1,comp2, num_features, get_feature(features[0]))
    elif num_features == 2:
        X,t = extract_features_label(train_df,comp1,comp2, num_features, get_feature(features[0]), get_feature(features[1]))
        x_test, t_test = extract_features_label(test_df,comp1,comp2, num_features, get_feature(features[0]), get_feature(features[1]))
    elif num_features == 3:
        X,t = extract_features_label(train_df,comp1,comp2, num_features, get_feature(features[0]), get_feature(features[1]), get_feature(features[2]))
        x_test, t_test = extract_features_label(test_df,comp1,comp2, num_features, get_feature(features[0]), get_feature(features[1]), get_feature(features[2]))
    elif num_features == 4:
        X,t = extract_features_label(train_df,comp1,comp2, num_features, get_feature(features[0]), get_feature(features[1]), get_feature(features[2]), get_feature(features[3]))
        x_test, t_test = extract_features_label(test_df,comp1,comp2, num_features, get_feature(features[0]), get_feature(features[1]), get_feature(features[2]), get_feature(features[3]))
        
        
    t = t.values.reshape([len(t),1])
    t_test = t_test.values.reshape([len(t_test),1])
    
    nn = NNetwork(features=num_features, hidden_layers=hidden_layers_arr, output_neurons=output_neurons, learning_rate=learning_rate)
    costs = nn.train(X, t, epochs=1000)
    y_pred = nn.predict(x_test)
    # evaluate the accuracy
    acc = accuracy(t_test, y_pred)
    print(acc)
    plt.plot(costs)
    plt.show()
    
    
    
    
        
        
    