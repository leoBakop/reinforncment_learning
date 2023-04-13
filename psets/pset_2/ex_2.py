import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


def mw_experts(df, start, T, best):
    """this function will implement the mw_experts, considering that there are
    as many experts as the number of servers, and every expert suggests to play the
    corresponding server. As a result to this, every expert i, except the right one is
    going to have loss in round t =load[i,t] - min(load[:,t])  and the right one is going to have loss = 0."""

    load = df
    k = load.shape[0]
    weights = np.ones(k)
    regret = np.zeros(T)
    best_score = np.zeros(T)
    alg_score = np.zeros(T)

    for t in range(start, T):
        #choosing the expert "based" on the weights
        sample = np.random.choice(weights,  size=1, replace=False, p=weights/np.sum(weights))
        expert = np.array(np.where(sample == weights))
        expert = expert[0][0] #in case that this weight exists more than one time
        #calculating the loss
        load_in_t = load.iloc[:,t]
        loss = load_in_t - np.min(load_in_t) #stores the loss per expert in round t 
        #loss = np.sum(load_in_t) - load_in_t[expert]
        weights = update_weights(weights, loss, k, T)
        #calculating the regret
        best_score[t] = best_score[t-1] + load_in_t[best] if t>0 else load_in_t[best]
        alg_score[t] = alg_score[t-1] + load_in_t[expert] if t>0 else load_in_t[expert]
        regret[t] = -(best_score[t] - alg_score[t])/(t+1) #the form of the graph (without the minus) is the beest score,
        #so the load of the best server, in round t, minus the load of the experts suggestion in round t.
        #As a result best_score <= alg_score and in convergance best_score == alg_score
    return weights, regret

def update_weights(weights, losses, k, T):
    eta = np.sqrt(np.log(k)/T)
    #new_weights = ((1-eta)**losses)*weights
    new_weights = np.multiply(np.power(1-eta, losses), weights)
    return np.array(new_weights)


def mw_bandits(df, start, T, best):
    
    load = df
    k = load.shape[0]
    gamma = np.sqrt(np.log(k)/T)
    weights = np.ones(k)
    p = np.ones(k)
    q = np.ones(k)
    regret = np.zeros(T)
    best_score = np.zeros(T)
    alg_score = np.zeros(T)
    for t in range(start, T):
        #calculating the propabilites
        p = weights/np.sum(weights)
        q = (1-gamma)*p + gamma/k

        #choosing the expert "based" on the weights
        sample = np.random.choice(weights,  size=1, replace=False, p=q)
        expert = np.array(np.where(sample == weights))
        expert = expert[0][0] #in case that this weight exists more than one time
        #calculating the loss
        load_in_t = load.iloc[:,t]
        loss = list([0 if i != expert else (load_in_t[expert]-np.min(load_in_t))/q[expert] for i in range(k)])
        weights = update_weights(weights, loss, k, T)
        #calculating the regret
        
        best_score[t] =best_score[t-1] + load_in_t[best] if t>0 else load_in_t[best]
        alg_score[t] = alg_score[t-1] + load_in_t[expert] if t>0 else load_in_t[expert]
        regret[t] = -(best_score[t] - alg_score[t])/(t+1) #the form of the graph (without the minus) is the beest score,
        #so the load of the best server, in round t, minus the load of the experts suggestion in round t.
        #As a result best_score <= alg_score and in convergance best_score == alg_score
    return weights, regret

def ucb (df, start, T, best):
    
    load = df
    k = load.shape[0]
    regret = np.zeros(T)
    best_score = np.zeros(T)
    alg_score = np.zeros(T)
    server_traffic = np.zeros(k) #server trafic deg=fines how "free" is a server, so it is 1-server_load
    server_counter = np.zeros(k) #stores the times that every server was selected 
    #variables in order to keep track of the regret
    regret = np.zeros(T)
    best_score = np.zeros(T)
    alg_score = np.zeros(T)
    #initial k rounds that explores every arm for the first time
    #in order to "initialize" the counter to 1
    load_in_t = load.iloc[:,0]
    for i in range(start,k):
        server_load = load_in_t[i]
        value = 1-server_load
        server_traffic, server_counter = update_for_ucb(server_traffic, server_counter, i, value)
        best_score[i] =best_score[i-1] + load_in_t[best] if i>0 else load_in_t[best]
        alg_score[i] = alg_score[i-1] + load_in_t[i] if i>0 else load_in_t[i]
        regret[i] = -(best_score[i] - alg_score[i])/(i+1)
    for i in range(k,T):
        index = arg_max_ucb(server_traffic, server_counter, i)
        load_in_t = load.iloc[:,i]
        server_load = load_in_t[index]
        value = 1-server_load
        server_traffic, server_counter = update_for_ucb(server_traffic, server_counter, index, value)
        best_score[i] =best_score[i-1] + load_in_t[best]
        alg_score[i] = alg_score[i-1] + load_in_t[index]
        regret[i] = -(best_score[i] - alg_score[i])/(i+1)
    return server_traffic, regret

def update_for_ucb(server_traffic, server_counter, index, value):
    server_traffic[index] = (server_counter[index]*server_traffic[index] + value)/(server_counter[index]+1) #calulates the new mean
    server_counter[index]+=1
    return server_traffic, server_counter


def arg_max_ucb(reward, q, t):
    """"
    this method just calculates the ucb
    of every arm and returns the index of the
    arm with the biggest ucb
    input: the list with the average reward per arm,
    the list with the counters that the algorith has "visit" 
    every arm 
    """
    ucb_array = []
    for r,n in zip(reward, q):
        ucb_array.append(r+np.sqrt(np.log(t)/n))
    return np.argmax(np.array(ucb_array)) 

def main():
    T = 7000
    start = 0
    df = pd.read_csv('psets/pset_2/Milano_timeseries.csv', header = None)
    
    
    partial_data = df.iloc[:, start:T-1]
    sum_of_load = np.array(partial_data.apply(np.sum, axis=1))
    best = np.argmin(sum_of_load)
    print(f"the best server based on the data is {best}")

    weights_experts, regret_experts = mw_experts(df, start, T, best)
    print(f"the best server based on experts algorithm is {np.argmax(weights_experts)} \
          with weight {np.amax(weights_experts)}") #the weight of each server is not associated with
    #the times that is going to be selected, because we calculate the loss per expert in every round
    weights_bandits, regret_bandits = mw_bandits(df, start, T, best)
    print(f"the best server based on bandits algorithm is {np.argmax(weights_bandits)} \
          with weight {np.amax(weights_bandits)}") 
    
    weights_ucb, regret_ucb = ucb(df, start, T, best)
    print(f"the best server based on ucb algorithm is {np.argmax(weights_ucb)} \
          with weight {np.amax(weights_ucb)}")
    #graphs section 

    reference = 0.001*np.sqrt(range(start,T)) - .1
    plt.figure(1)
    plt.title("Regret comparison between both algorithms") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(start+1,T+1),regret_experts, label="MW expert  regret")
    plt.plot(np.arange(start+1,T+1),regret_bandits, label="MW bandits regret")
    plt.plot(np.arange(start+1,T+1),regret_ucb, label="Ucb regret")  
    #plt.plot(np.arange(start+1,T+1),reference, label="O(np.sqrt(T)) regret") 
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()