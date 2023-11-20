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

def extract_features_label(df: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.Series]:
    # Filter the dataframe to include only Setosa and Virginica rows
    filtered_df1 = df[df['variety'].isin(['Versicolor', 'Virginica'])]
    filtered_df2 = df[df['variety'].isin(['Setosa', 'Virginica'])]
    feature_df_dict = {
        1: filtered_df1,
        2: filtered_df2,
    }
    filtered_df = feature_df_dict[index]
    # Extract the required features and labels from the filtered dataframe
    features1 = filtered_df[['sepal.length', 'sepal.width']]
    features4 = filtered_df[['petal.length', 'petal.width']]
    features_dict ={
        1: features1,
        2: features4,
    }

    labels1 = filtered_df['variety'].map({'Versicolor': 0, 'Virginica': 1})
    labels2 = filtered_df['variety'].map({'Setosa': 0, 'Virginica': 1})
    labels_dict = {
        1: labels1,
        2: labels2,
    }
    return features_dict[index], labels_dict[index]


def extract_features_labels_NNET4(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    filtered_df = df[df['variety'].isin(['Setosa', 'Virginica'])]
    features = filtered_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    labels = filtered_df['variety'].map({'Setosa': 0, 'Virginica': 1})
    return features, labels

def extract_features_labels_NNET4_2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    filtered_df = df[df['variety'].isin(['Versicolor', 'Virginica'])]
    features = filtered_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    labels = filtered_df['variety'].map({'Versicolor': 0, 'Virginica': 1})
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

class NNet3:
    def __init__(self, features, hidden_neurons, output_neurons, learning_rate):
        self.features = features
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        
        # initialize weights
        
        # nueron layer of the hidden layer
        self.V = np.random.randn(self.features, self.hidden_neurons)
        #Creating a middle third layer whihc can take the shape of of H1.dot(self.U) + self.U0 this is beacuse thsi will haev alarge shaope than the input layer so requires having both a bias layer and weight layer equal to teh nuerons as apposed to the features.
        self.U = np.random.randn(self.hidden_neurons, self.hidden_neurons)
        # nueron layer of teh output layer
        self.W = np.random.randn(self.hidden_neurons, self.output_neurons)
        
        # initialize biases: 0
        self.U0 = np.zeros((self.hidden_neurons))
        self.V0 = np.zeros((self.hidden_neurons))
        self.W0 = np.zeros((self.output_neurons))
    
    def train(self, X, t, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # forward pass big guess
            
            #$ First hidden layer
            net_u = X.dot(self.V) + self.V0
            H1 = sigmoid(net_u)

            # New second hidden layer
            #This is terh amin idea duiffrence as now we are passing the output of teh first klayer into this secodn layer as opposed to the input layer
            net_v = H1.dot(self.U) + self.U0
            H2 = sigmoid(net_v)

            
            # Output layer
            net_z = H2.dot(self.W) + self.W0
            O = sigmoid(net_z)
            # this is the derivative of the cost function
            # backpropagation pass
            
            error_output = O - t
            d_W = H2.T.dot(error_output * function_derivative(net_z))
            d_W0 = np.sum(error_output * function_derivative(net_z), axis=0)

            # Corrected second hidden layer error propagation
            error_hidden_layer2 = error_output.dot(self.W.T) * function_derivative(net_v)
            d_U = H1.T.dot(error_hidden_layer2)
            d_U0 = np.sum(error_hidden_layer2, axis=0)

            # Corrected first hidden layer error propagation
            error_hidden_layer1 = error_hidden_layer2.dot(self.U.T) * function_derivative(net_u)
            d_V = X.T.dot(error_hidden_layer1)
            d_V0 = np.sum(error_hidden_layer1, axis=0)
          
            # # This is output layer updates
          
            # # This will take 
            # d_W = H2.T.dot(error_output * function_derivative(net_z))
            # # this is the derivative of the cost function with respect to the bias which will update the bias
            # d_W0 = np.sum(error_output * function_derivative(net_z), axis=0)
            # # this is the derivative of the cost function with respect to the weights which will update the weights

            # # d_w is the derivative of the cost function with respect to the weights
            # # d_wo is the derivative of the cost function with respect to the biases
            
            
            
            # #This is hidden layer updates
            
            # #new hidden layer
            

            
            
            # #in a nueral network you update both the weights and the baises during each pass of the backpropagation
            # error_hidden_layer2 = error_output.dot(self.W.T) * function_derivative(net_u)
            # # this is the derivative of the cost function with respect to the weights which will update the weights
            # d_V = X.T.dot(error_hidden_layer2)
            # # this is the derivative of the cost function with respect to the bias which will update the bias
            # d_V0 = np.sum(error_hidden_layer2, axis=0)
            
            # error_hidden_layer1 = error_hidden_layer2.dot(self.W.T) * function_derivative(net_v)
            # d_U = H1.T.dot(error_hidden_layer1)
            # d_UO = np.sum(error_hidden_layer1, axis=0)
            
            
            
            
            
            
            # update weights and biases
            self.W -= self.learning_rate * d_W
            self.W0 -= self.learning_rate * d_W0
            self.V -= self.learning_rate * d_V
            self.V0 -= self.learning_rate * d_V0
            self.U -= self.learning_rate * d_U
            self.U0 -= self.learning_rate * d_U0
            
            #find the cost function
            if epoch % 10 == 0:
                loss =  np.square(np.subtract(t,O)).mean() 
                costs.append(loss)
            # if epoch %50 == 0:
            #     print("Epoch: %d, cost: %.4f" % (epoch, loss))
                
        return costs
    
    def predict(self, X):
        net_u = X.dot(self.V) + self.V0
        H1 = sigmoid(net_u)
        net_v = H1.dot(self.U) + self.U0
        H2 = sigmoid(net_v)
        net_z = H2.dot(self.W) + self.W0
        O = sigmoid(net_z)
        return (O > 0.5).astype(int)
    





class NNet2:
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
        
    def accuracy(t, y_pred):
        accuracy = np.sum(t == y_pred) / len(t)
        return accuracy
    
    def train(self, X, t, epochs=1000):
        costs = []
        accuracy = []
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
                print(" %d & %.4f &" % (epoch, loss))
                
        return costs
    
    def predict(self, X):
        net_u = X.dot(self.V) + self.V0
        H = sigmoid(net_u)
        net_z = H.dot(self.W) + self.W0
        O = sigmoid(net_z)
     
        return (O > 0.5).astype(int)
    

def create_overleaf_table(epoch_costs: list[list[Tuple[int, float]]], feature_sets: list[str], filename: str):
    with open(filename, 'w') as f:
        f.write("\\hline\n")
        f.write(" & ".join([f"{feature_con[fs]} & " for fs in feature_sets]) + "\\\\\n")
        f.write("\\hline\n")
        f.write("Epoch & Cost & " * len(feature_sets) + "\\\\\n")
        f.write("\\hline\n")
        max_epochs = max([len(ec) for ec in epoch_costs])
        for i in range(max_epochs):
            row = []
            for ec in epoch_costs:
                if i < len(ec):
                    epoch, cost = ec[i]
                    # Ensure epoch and cost are single numeric values
                    epoch = int(epoch)  # Convert to int if not already
                    cost = float(cost)  # Convert to float if not already
                    row.append(f"{epoch} & {cost:.4f}")
                else:
                    row.append("&")
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\hline\n")

if __name__ == "__main__":
    def accuracy(t, y_pred):
        accuracy = np.sum(t == y_pred) / len(t)
        return accuracy
    
    train_df = read_data("iris_training_data.csv")
    test_df = read_data("iris_testing_data.csv")

    feature_combinations = [
        ['sepal.length', 'sepal.width'],
        ['petal.length', 'petal.width']
    ]
    
    feature_con ={
        'sepal.length x sepal.width': 'SL X SW',
        'sepal.length x petal.width': 'SL X PW',
        'petal.length x sepal.width': 'PL X SW',
        'petal.length x petal.width': 'PL X PW',
    }
    
    # epoch_costs_all = []
    # index = 0
    # for features in feature_combinations:
    #     print(f"Training on features: {features}")
    #     if features[0] == 'sepal.length' and features[1] == 'sepal.width':
    #         index = 1
    #     elif features[0] == 'petal.length' and features[1] == 'petal.width':
    #         index = 2
    #     X, t = extract_features_label(train_df, index)
    #     X_test, t_test = extract_features_label(test_df, index)

    #     t = t.values.reshape([len(t), 1])
    #     t_test = t_test.values.reshape([len(t_test), 1])
        
    #     #NNET1

    #     nn = NNet2(features=2, hidden_neurons=20, output_neurons=1, learning_rate=0.01)
    #     cost = nn.train(X, t, epochs=1000)  # Adjust epochs as needed
    #     y_pred = nn.predict(X_test)
    #     acc = accuracy(t_test, y_pred)
    #     print(f"Accuracy: {acc}")

        
    #     epoch_costs = [(epoch * 10, cost) for epoch, cost in enumerate(cost) if epoch % 5 == 0]  # Assuming costs are recorded every 50 epochs
    #     epoch_costs_all.append(epoch_costs)

    # create_overleaf_table(epoch_costs_all, [f"{f[0]} x {f[1]}" for f in feature_combinations], "comparison_table_NNET1.txt")
    
    # epoch_costs_all = []
    # index = 0
    # for features in feature_combinations:
    #     print(f"Training on features: {features}")
    #     if features[0] == 'sepal.length' and features[1] == 'sepal.width':
    #         index = 1
    #     elif features[0] == 'petal.length' and features[1] == 'petal.width':
    #         index = 2
    #     X, t = extract_features_label(train_df, index)
    #     X_test, t_test = extract_features_label(test_df, index)

    #     t = t.values.reshape([len(t), 1])
    #     t_test = t_test.values.reshape([len(t_test), 1])

    #     #NNET2

    #     nn = NNet2(features=2, hidden_neurons=20, output_neurons=1, learning_rate=0.01)
    #     cost = nn.train(X, t, epochs=1000)  # Adjust epochs as needed
    #     y_pred = nn.predict(X_test)
    #     acc = accuracy(t_test, y_pred)
    #     print(f"Accuracy: {acc}")

    #     epoch_costs = [(epoch * 10, cost) for epoch, cost in enumerate(cost) if epoch % 5 == 0]  # Assuming costs are recorded every 50 epochs
    #     epoch_costs_all.append(epoch_costs)

    # create_overleaf_table(epoch_costs_all, [f"{f[0]} x {f[1]}" for f in feature_combinations], "comparison_table_NNET2.txt")

    # epoch_costs_all = []
    # index = 0
    # for features in feature_combinations:
    #     print(f"Training on features: {features}")
    #     if features[0] == 'sepal.length' and features[1] == 'sepal.width':
    #         index = 1
    #     elif features[0] == 'petal.length' and features[1] == 'petal.width':
    #         index = 2
    #     X, t = extract_features_label(train_df, index)
    #     X_test, t_test = extract_features_label(test_df, index)

    #     t = t.values.reshape([len(t), 1])
    #     t_test = t_test.values.reshape([len(t_test), 1])

        
        
        
    #     #NNET3
        
        
    #     nn = NNet3(features=2, hidden_neurons=5, output_neurons=1, learning_rate=0.01)
    #     cost = nn.train(X, t, epochs=1000)  # Adjust epochs as needed
    #     y_pred = nn.predict(X_test)
    #     acc = accuracy(t_test, y_pred)
    # #     print(f"Accuracy: {acc}")
    #     plt.plot(cost)
    #     plt.title(f"Cost for {features}")
    #     plt.show()

    #     epoch_costs = [(epoch * 10, cost) for epoch, cost in enumerate(cost) if epoch % 5 == 0]  # Assuming costs are recorded every 50 epochs
    #     epoch_costs_all.append(epoch_costs)

    # create_overleaf_table(epoch_costs_all, [f"{f[0]} x {f[1]}" for f in feature_combinations], "comparison_table_NNET3.txt")
    
    X, t =  extract_features_labels_NNET4(train_df)
    X_test, t_test = extract_features_labels_NNET4(test_df)
    t = t.values.reshape([len(t),1])
    t_test = t_test.values.reshape([len(t_test),1])
    
    
    
    
    #NNET4 on data set 1    
    nn = NNet2(features=4, hidden_neurons=5, output_neurons=1, learning_rate=0.01)
    cost = nn.train(X, t, epochs=1000)
    y_pred = nn.predict(X_test)
    # evaluate the accuracy
    acc = accuracy(t_test,y_pred)
    print(acc)
    plt.plot(cost)
    plt.show()
    
    X, t =  extract_features_labels_NNET4_2(train_df)
    X_test, t_test = extract_features_labels_NNET4_2(test_df)
    t = t.values.reshape([len(t),1])
    t_test = t_test.values.reshape([len(t_test),1])
    
    # create Neural Network class
    
    
    
    #NNET4 on data set 1    
    nn = NNet2(features=4, hidden_neurons=5, output_neurons=1, learning_rate=0.01)
    cost = nn.train(X, t, epochs=1000)
    y_pred = nn.predict(X_test)
    # evaluate the accuracy
    acc = accuracy(t_test,y_pred)
    print(acc)
    plt.plot(cost)
    plt.show()
