import numpy as np
import pandas as pd
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1.0-x)


def update_weights(num_inputs, num_hidden, num_outputs, update=True):
    if update is False:
        input_range = 1.0 / np.sqrt(num_inputs)
        output_range = 1.0 / np.sqrt(num_hidden)

        inputweights = np.random.normal(loc=0, scale=input_range, size=(num_inputs, num_hidden))
        outputweights = np.random.normal(loc=0, scale=output_range, size=(num_hidden, num_outputs))
        return inputweights, outputweights


def initialize_hidden_nodes(num_hidden):
    hidden_nodes = [1.0] * 10
    return hidden_nodes

def update_activation(num_inputs, num_hidden, num_outputs, update = True):
    if update is False:
        input_activation = [1.0] * num_inputs
        hidden_activation = [1.0] * num_hidden
        output_activation = [1.0] * num_outputs
        return input_activation, hidden_activation, output_activation


def feedforward(ia, ha, oa, inpweight, outpweight, inputs, outputs, hiddens):
    #Input activations fill
    inp_incrementer = 0
    for i in range(len(ia)-1):
        ia[i] = inputs[i]
        inp_incrementer += 1
    #Hidden activations fill
    for j in range(len(ha)):
        sum = 0.0
        for i in range(len(ia)):
            sum += ia[i] * inpweight[i][j]
        ha[j] = sigmoid(sum)
    #output activations fill
    for k in range(len(oa)):
        sum = 0.0
        for j in range(len(ha)):
            sum += ha[j] + outpweight[j][k]
        oa[k] = sigmoid(sum)
    return oa[:]


def backPropagate(actual, ia, ha, oa, inpweight, outpweight, inputs, outputs, hiddens, learning_rate, inpmagchange, outpmagchange):
    output_differentials = [0.0] * len(oa)
    for k in range(len(oa)):
        error_amnt = -(actual[k] - oa[k])
        output_differentials[k] = sigmoid_derivative(oa[k]) * error_amnt

    hidden_differential = [0.0] * len(ha)
    for j in range(len(ha)):
        error = 0.0
        for k in range(len(oa)):
            error += output_differentials[k] * outpweight[j][k]
        hidden_differential[j] = sigmoid(ha[j]) * error
    #update weights hidden->output
    for j in range(len(ha)):
        for k in range(len(oa)):
            change = output_differentials[k] * ha[j]
            outpweight[j][k] -= learning_rate * change[k] + outpmagchange[j][k]
            outpmagchange[j][k] = change[k]
    for i in range(len(ia)):
        for j in range(len(ia)):
            change = hidden_differential[j] * ia[i]
            inpweight[i][j] -= learning_rate * change[j] + inpmagchange[i][j]
            inpweight[i][j] = change[j]

    error = 0.0
    for k in range(len(actual)):
        error += 0.5 * (actual[k] - oa[k]) ** 2
    return error


def fit(actual, iterations, ia, ha, oa, inpweight, outpweight, inputs, outputs, hiddens, learning_rate, inpmagchange, outpmagchange):
    datapoints = inputs
    for i in range(iterations):
        error = 0.0
        random.shuffle(datapoints)
        for p in range(len(datapoints)):
            feedforward(ia, ha, oa, inpweight, outpweight, inputs[p], outputs[p], hiddens)
            error += backPropagate(actual, ia, ha, oa, inpweight, outpweight, inputs, outputs, hiddens, learning_rate, inpmagchange, outpmagchange)
        with open('error.txt', 'a') as erroroutput:
            erroroutput.write(str(error) + '\n')
            erroroutput.close()
        if i% 100 == 0:
            print('error %-.5f' %error)


def predict(inputs, ia, ha, oa, inpweight, outpweight, outputs, hiddens):
    predictions = []
    for i in inputs:
        predictions.append(feedforward(ia, ha, oa, inpweight, outpweight, inputs, outputs, hiddens))
    return predictions

def main():
    dataset = pd.read_csv('iris.data.csv', header=None)  # Load dataset into Pandas
    inputs = dataset.drop(4, axis=1).as_matrix()  # Numpy array of input data
    y = dataset[4]  # Load Iris flowers into panda series
    outputs = pd.get_dummies(y).as_matrix()  # Numpy array of output data encoded to dummy variables
    hiddens = initialize_hidden_nodes(10)
    bias = np.ones(inputs.shape[0])
    real_inputs = np.vstack((inputs.T, bias))
    real_inputs = real_inputs.T
    input_weights, outputweights = update_weights(inputs.shape[1], len(hiddens), outputs.shape[1], False)
    input_activation, hidden_activation, output_activation = update_activation(inputs.shape[1], len(hiddens), outputs.shape[1], False)
    #oa = feedforward(input_activation,hidden_activation, output_activation, input_weights, outputweights, real_inputs, outputs, hiddens)
    inp_weight_magnitude = np.zeros((inputs.shape[1], len(hiddens)))
    outp_weight_magnitude = np.zeros((len(hiddens), outputs.shape[1]))
    fit(outputs, 10000, input_activation, hidden_activation, output_activation,input_weights, outputweights, real_inputs, outputs, hiddens, .5,inp_weight_magnitude, outp_weight_magnitude)
main()
























