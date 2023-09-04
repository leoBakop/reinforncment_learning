import numpy as np
import matplotlib.pyplot as plt

horizon = int(1_500_000/250_000)
file_path = "DQN_Agent/data/dropping_lr/threshold_False_per_False_0001_drop.npy"
# Load the CSV file into a NumPy array
drop= np.load(file_path, allow_pickle=True)


#file_path = "DQN_Agent/data/lr_experiments/threshold_False_per_False_001.npy"
#agent_001 = np.load(file_path, allow_pickle=True)

file_path = "DQN_Agent/data/lr_experiments/threshold_False_per_False_0001.npy"
agent_0001 = np.load(file_path, allow_pickle=True)

file_path = "DQN_Agent/data/lr_experiments/threshold_False_per_False_00001.npy"
agent_00001 = np.load(file_path, allow_pickle=True)


plt.figure(1)
plt.xlabel("Round T") 
plt.ylabel("reward in t") 
plt.plot(np.arange(1,horizon+1),drop, label="learning rate reduction")
#plt.plot(np.arange(1,horizon+1),agent_001, label="lr = 10**(-3)")
plt.plot(np.arange(1,horizon+1),agent_0001, label="lr = 10**(-4)")
plt.plot(np.arange(1,horizon+1),agent_00001, label="lr = 10**(-5)")
plt.grid()
plt.legend()
plt.savefig("./DQN_Agent/images/report/lr_comparison_reduction") # I would have to save fig in order to store it. That is working without depending on .show
plt.show()
print("end")