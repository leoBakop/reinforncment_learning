import numpy as np 
import random
import matplotlib.pyplot as plt 

def create_ranges(k, classing):
    """
    this method just create the reward range for every arm.
    those ranges are randomly generated.
    the function guarantees that every range is consist of a lower and an upper bound
    """
    ranges = [] #create a numpy array with len = k
    for i in range(k):
        lower = np.random.random()
        upper = np.random.random()
        (lower,upper) = (upper, lower) if upper < lower else (lower, upper)
        e = .1 
        while ((upper+lower)/2 > e and  (upper+lower)/2 < 1-e and classing):
            lower = np.random.random()
            upper = np.random.random()
            (lower,upper) = (upper, lower) if upper < lower else (lower, upper)
        
        ranges.append(np.array([lower, upper]))


    return np.array(ranges) #to cooment the 10*np.arr

def pull_arm(min, max):
    return random.uniform(min, max)

def epsilon_greedy(T, arms, best, k):
    eps = 1
    dt = .0001
    arms_score = np.zeros(len(arms)) #stores the mean per arm
    arms_counter = np.zeros(len(arms)) #stores the times that every arm was pulled 
    #variables in order to keep track of the regret
    regret = np.zeros((T))
    best_score = np.zeros((T))
    alg_score = np.zeros((T))
    for t in range(T):
        eps = (1/((t+dt)**(1/3)))*((k*np.log(t+dt))**(1/3))
        p = random.random()
        if eps < p: #pull an random arm
            index = random.randint(0, len(arms)-1 )
            (arms_score, arms_counter, value) = pull_and_update(arms, index, arms_score, arms_counter)
        else : #pull the arm with the best mean
            index = np.argmax(arms_score)
            (arms_score, arms_counter, value) = pull_and_update(arms, index, arms_score, arms_counter)
        #calculating the regret
        best_score[t] =best_score[t-1]+ best if t>0 else best
        alg_score[t] = alg_score[t-1]+ value if t>0 else value
        regret[t] = (best_score[t] - alg_score[t])/(t+1)
    return np.argmax(arms_score), regret, alg_score

def ucb(T, arms, k, best):
    arms_score = np.zeros(len(arms)) #stores the mean per arm
    arms_counter = np.zeros(len(arms)) #stores the times that every arm was pulled 
    #variables in order to keep track of the regret
    regret = np.zeros(T)
    best_score = np.zeros(T)
    alg_score = np.zeros(T)
    #initial k rounds that explores every arm for the first time
    #in order to "initialize" the counter to 1
    for i in range(k):
        (arms_score, arms_counter, value) = pull_and_update(arms, i, arms_score, arms_counter)
        best_score[i] =best_score[i-1] + best if i>0 else best
        alg_score[i] = alg_score[i-1] + value if i>0 else value
        regret[i] = (best_score[i] - alg_score[i])/(i+1)
    for i in range(k,T):
        index = arg_max_ucb(arms_score, arms_counter, i)
        (arms_score, arms_counter, value) = pull_and_update(arms, index, arms_score, arms_counter)
        best_score[i] =best_score[i-1] + best
        alg_score[i] = alg_score[i-1] + value
        regret[i] = (best_score[i] - alg_score[i])/(i+1)

    return np.argmax(arms_score), regret, alg_score

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

def pull_and_update(arms, index, arms_score, arms_counter):
    arm = arms[index]
    (min, max)=arm[:]
    value = pull_arm(min, max)
    arms_score[index] = (arms_counter[index]*arms_score[index] + value)/(arms_counter[index]+1) #calulates the new mean
    arms_counter[index]+=1
    return (arms_score, arms_counter, value)

def print_list(l):
    for i,j in enumerate(l):
        print(f"{i}th element is {j}")

def main():

    #creating the environment
    k = 10
    T = k*100

    #initialize/defining the range of rewards per arm
    #and calculate the best arm
    bandit = create_ranges(k, False)
    means = map(lambda x: (x[0]+x[1])/2, bandit)
    means = list(means)
    best = np.argmax(np.array(means))
    ideal_reward=list([i*means[best] for i in range(T)])
    #apply the two algorithm
    best_arm_egreedy, eps_greedy_regret, eps_alg_score = epsilon_greedy(T, bandit, means[best], k)
    best_arm_ucb, ucb_regret, ucb_alg_score = ucb(T, bandit, k, means[best])
    
    #printing/plotting results
    print_list(list(means))
    print(f"best arm is {best}")
    print(f"best arm based on e greedy is {best_arm_egreedy}")
    print(f"best arm based on ucb is {best_arm_ucb}")


    #graphs section 
    plt.figure(1)
    plt.title("Regret comparison between both algorithms") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(1,T+1),eps_greedy_regret, label="eps-greedy regret") 
    plt.plot(np.arange(1,T+1),ucb_regret, label="ucb regret") 
    plt.legend()
    plt.show()

    #
    plt.figure(2)
    plt.title("Score comparison between both algorithms") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(1,T+1),eps_alg_score, label="eps-greedy score") 
    plt.plot(np.arange(1,T+1),ucb_alg_score, label="ucb score")
    plt.plot(np.arange(1,T+1),ideal_reward, label="ideal score") 
 
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()


