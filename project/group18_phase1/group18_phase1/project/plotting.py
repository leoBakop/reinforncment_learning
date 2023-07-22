import numpy as np
import matplotlib.pyplot as plt

horizon = 80_000
r_q = np.loadtxt("./data/rewards/q_learning_True_threshold_True_aggressive_False.csv", delimiter=",",dtype = float)
r_p = np.loadtxt("./data/rewards/q_learning_False_threshold_True_aggressive_False.csv", delimiter=",",dtype = float)


plt.figure(1)
plt.title(f" Agent's Reward (Against tight opponent)") 
plt.xlabel("Round T") 
plt.ylabel("Total Score") 
plt.plot(np.arange(1,horizon+1),r_q, 'r--', label="Cumulative Reward (Q-agent)")
plt.plot(np.arange(1,horizon+1),r_p, 'b--', label="Cumulative Reward (Policy Iteration-agent)")   
plt.grid()
plt.legend()
plt.savefig(f'images/aggregate/defensive.jpg')
plt.show()


r_q = np.loadtxt("./data/rewards/q_learning_True_threshold_True_aggressive_True.csv", delimiter=",",dtype = float)
r_p = np.loadtxt("./data/rewards/q_learning_False_threshold_True_aggressive_True.csv", delimiter=",",dtype = float)


plt.figure(2)
plt.title(f" Agent's Reward (Against loose opponent)") 
plt.xlabel("Round T") 
plt.ylabel("Total Score") 
plt.plot(np.arange(1,horizon+1),r_q, 'r--', label="Cumulative Reward (Q-agent)")
plt.plot(np.arange(1,horizon+1),r_p, 'b--', label="Cumulative Reward (Policy Iteration-agent)")   
plt.grid()
plt.legend()
plt.savefig(f'images/aggregate/offensive.jpg')
plt.show()


r_q = np.loadtxt("data/rewards/q_learning_True_threshold_False_aggressive_False.csv", delimiter=",",dtype = float)
r_p = np.loadtxt("data/rewards/q_learning_False_threshold_False_aggressive_False.csv", delimiter=",",dtype = float)


plt.figure(3)
plt.title(f" Agent's Reward (Against Random opponent)") 
plt.xlabel("Round T") 
plt.ylabel("Total Score") 
plt.plot(np.arange(1,horizon+1),r_q, 'r--', label="Cumulative Reward (Q-agent)")
plt.plot(np.arange(1,horizon+1),r_p, 'b--', label="Cumulative Reward (Policy Iteration-agent)")   
plt.grid()
plt.legend()
plt.savefig(f'images/aggregate/random.jpg')
plt.show()