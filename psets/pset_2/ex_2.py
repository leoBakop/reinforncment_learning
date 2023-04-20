import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


def mw_experts(df, start, T, best, display = True):
    """this function will implement the mw_experts, considering that there are
    as many experts as the number of servers, and every expert suggests to play the
    corresponding server. As a result to this, every expert i, except the right one is
    going to have loss in round t =load[i,t] - min(load[:,t])  and the right one is going to have loss = 0."""

    load = df
    k = load.shape[0]
    weights = np.ones(k)
    regret = np.zeros(T-start)
    best_score = np.zeros(T-start)
    alg_score = np.zeros(T-start)

    for t in range(start, T):
        #choosing the expert "based" on the weights
        sample = np.random.choice(weights,  size=1, replace=False, p=weights/np.sum(weights)) #sample returns a weight
        expert = np.array(np.where(sample == weights)) #gettin the index of the weight
        expert = expert[0][0] #in case that this weight exists more than one time
        #calculating the loss
        load_in_t = load.iloc[:,t] #storing the loads in that round
        loss = load_in_t - np.min(load_in_t) #stores the loss per expert in round t 
        weights = update_weights(weights, loss, k, T) #update the weights based on theory
        #calculating the regret
        t=t-start
        best_score[t] = best_score[t-1] + load_in_t[best] if t>0 else load_in_t[best]
        alg_score[t] = alg_score[t-1] + load_in_t[expert] if t>0 else load_in_t[expert]
        regret[t] = -(best_score[t] - alg_score[t])/(t+1) #minus stands in order to keep the regret positive
        #but it has also mathematical sence because now the regret is not "regret of rewards"
        #but "regret of losses"
        if t%1000 == 0 and display:
            print(f'best server base on experts is {expert} and best server is {np.argmin(load_in_t)} in round {t}')
    return weights, regret

def update_weights(weights, losses, k, T):
    """this method updates the weights as theory suggests.
       there is common in both algorithms. The difference
       is included in the loss input. 
    """
    eta = np.sqrt(np.log(k)/T) #defining eta as theory suggests
    new_weights = np.multiply(np.power(1-eta, losses), weights)
    return np.array(new_weights)


def mw_bandits(df, start, T, best,display = True, with_q = True):
    
    load = df
    k = load.shape[0]
    gamma = np.sqrt(np.log(k)/T)
    weights = np.ones(k)
    p = np.ones(k)
    q = np.ones(k)
    regret = np.zeros(T-start)
    best_score = np.zeros(T-start)
    alg_score = np.zeros(T-start)
    for t in range(start, T):
        #calculating the propabilites
        p = weights/np.sum(weights)
        q = (1-gamma)*p + gamma/k if with_q else p #this line was written only for experiment purposes
        #the default algorithm is the q when with_q = True (that's wsy was set as default)

        #choosing the expert "based" on the weights, just like the previous algorithm
        sample = np.random.choice(weights,  size=1, replace=False, p=q)
        expert = np.array(np.where(sample == weights))
        expert = expert[0][0] #in case that this weight exists more than one time
        #calculating the loss
        load_in_t = load.iloc[:,t]
        if with_q: #again this if exists only for experimental resons. The correct is when with_q = True
            loss = list([0 if i != expert else (load_in_t[expert]-np.min(load_in_t))/q[expert] for i in range(k)]) #updating the loss of only the choosen server
        else:
            loss = list([0 if i != expert else (load_in_t[expert]-np.min(load_in_t)) for i in range(k)])
        weights = update_weights(weights, loss, k, T)
        #calculating the regret
        t=t-start
        best_score[t] =best_score[t-1] + load_in_t[best] if t>0 else load_in_t[best]
        alg_score[t] = alg_score[t-1] + load_in_t[expert] if t>0 else load_in_t[expert]
        regret[t] = -(best_score[t] - alg_score[t])/(t+1)#minus stands in order to keep the regret positive
        #but it has also mathematical sence because now the regret is not "regret of rewards"
        #but "regret of losses" 
        if t%1000 == 0 and display:
            print(f'best server based on bandits is {expert} and best server is {np.argmin(load_in_t)} in round {t}')
    return weights, regret

def ucb (df, start, T, best, display = True):
    """method is implemented just like the previous pset"""
    load = df
    k = load.shape[0]
    regret = np.zeros(T)
    best_score = np.zeros(T)
    alg_score = np.zeros(T)
    server_traffic = np.zeros(k) #server trafic defines how "free" is a server, so it is 1-server_load
    server_counter = np.zeros(k) #stores the times that every server was selected 

    #variables in order to keep track of the regret
    regret = np.zeros(T-start)
    best_score = np.zeros(T-start)
    alg_score = np.zeros(T-start)
    #initial k rounds that explores every arm for the first time
    #in order to "initialize" the counter to 1
    
    for i in range(start,k+start):
        #I select sequential all the severs for one episode
        i-=start
        load_in_t = load.iloc[:,i] 
        server_load = load_in_t[i]
        value = -server_load #ucb will predict the reward
        server_traffic, server_counter = update_for_ucb(server_traffic, server_counter, i, value) #update the the ucb components
        #calulating regret
        best_score[i] =best_score[i-1] + load_in_t[best] if i>0 else load_in_t[best]
        alg_score[i] = alg_score[i-1] + load_in_t[i] if i>0 else load_in_t[i]
        regret[i] = -(best_score[i] - alg_score[i])/(i+1)
    #main loop of the algorithm
    for i in range(k+start,T):
        index = arg_max_ucb(server_traffic, server_counter, i) #expert = argmax(ucb_i)
        load_in_t = load.iloc[:,i]
        server_load = load_in_t[index]
        value = -server_load
        server_traffic, server_counter = update_for_ucb(server_traffic, server_counter, index, value) #update the the ucb components
        i-=start
        #calculating regert
        best_score[i] =best_score[i-1] + load_in_t[best]
        alg_score[i] = alg_score[i-1] + load_in_t[index]
        regret[i] = -(best_score[i] - alg_score[i])/(i+1) if -(best_score[i] - alg_score[i])/(i+1) < .5 else .5
        if i%1000 == 0 and display:
            print(f'best server based on ucb is {index} and best server is {np.argmin(load_in_t)} in round {i}')
    return server_traffic, regret

def update_for_ucb(server_traffic, server_counter, index, value):
    """method that updates and return the new components"""
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
    r = False #variable for experiments, default = False
    if r :
        T = int(7000*np.random.rand())
        start = int(1000*np.random.rand())
        if start >= T: return 0 
    else :
        T=7000
        start = 0 
    
    df = pd.read_csv('./pset_2/Milano_timeseries.csv', header = None)
    display = True #in order to show the selected server every 10^3 rounds
    
    #calculating the best server, considering that best is the on with the lower sum of loads
    partial_data = df.iloc[:, start:T-1]
    sum_of_load = np.array(partial_data.apply(np.sum, axis=1))
    best = np.argmin(sum_of_load)
    print(f"the best server based on the data is {best}")
    print("--------------experts------------------")
    weights_experts, regret_experts = mw_experts(df, start, T, best, display)
    print(f"the best server based on experts algorithm is {np.argmax(weights_experts)} \
          with weight {np.amax(weights_experts)}") #the weight of each server is not associated with
    #the times that is going to be selected, because we calculate the loss per expert in every round
    #that's why is constant for every time that this programm will be executed
    print("---------------bandits-----------------")
    weights_bandits, regret_bandits = mw_bandits(df, start, T, best, display=display, with_q = True)
    print(f"the best server based on bandits algorithm is {np.argmax(weights_bandits)} \
          with weight {np.amax(weights_bandits)}")#the weight of each server depends on the times that is going to be calculated
    #thats why there are slightly changes between each execution
    print("---------------ucb-----------------")
    weights_ucb, regret_ucb = ucb(df, start, T, best, display)
    print(f"the best server based on ucb algorithm is {np.argmax(weights_ucb)} \
          with weight {np.amax(weights_ucb)}") #in this case wights_ucb are not some kinf of weights
    #but it was kept for consistency
    
    #graphs section 
    plt.figure(1)
    plt.title(f"Regret comparison between algorithms where start ={start} end ={T}") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(start+1,T+1),regret_experts, label="MW expert  regret")
    plt.plot(np.arange(start+1,T+1),regret_bandits, label="MW bandits regret")
    plt.plot(np.arange(start+1,T+1),regret_ucb, label="Ucb regret")  
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()