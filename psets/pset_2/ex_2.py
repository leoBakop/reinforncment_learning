import pandas as pd
import numpy as np

def mw_experts(df, start, T):
    """this function will implement the mw_experts, considering that there are
    as many experts as the number of servers, and every expert suggests to play the
    corresponding server. As a result to this, every expert i, except the right one is
    going to have loss in round t =load[i,t] - min(load[:,t])  and the right one is going to have loss = 0."""

    load = df
    k = load.shape[0]
    weights = np.ones(k) 
    for t in range(start, T):
        #choosing the expert "based" on the weights
        sample = np.random.choice(weights,  size=1, replace=False, p=weights/np.sum(weights))
        expert = np.array(np.where(sample == weights))
        expert = expert[0][0] #in case that this weight exists more than one time
        #calculating the loss
        load_in_t = load.iloc[:,t]
        loss = load_in_t - np.min(load_in_t) #stores the loss per expert in round t 
        weights = update_weights_experts(weights, loss, k, T)
    return weights

def update_weights_experts(weights, losses, k, T):
    eta = np.sqrt(np.log(k)/T)
    new_weights = ((1-eta)**losses)*weights
    new_weights = np.multiply(np.power(1-eta, losses), new_weights)
    return np.array(new_weights)

if __name__ == '__main__':
    T = 7000
    start = 0
    df = pd.read_csv('psets/pset_2/Milano_timeseries.csv', header = None)
    weights = mw_experts(df, start, T)
    print(f"the best server based on algorithm is {np.argmax(weights)} \
          with weight {np.amax(weights)}") #the weight of each server is not associated with
    #the times that is going to be selected, because we calculate the loss per expert in every round
    
    partial_data = df.iloc[:, start:T-1]
    sum_of_load = np.array(partial_data.apply(np.sum, axis=1))
    print(f"the best server based on the data is {np.argmin(sum_of_load)}")