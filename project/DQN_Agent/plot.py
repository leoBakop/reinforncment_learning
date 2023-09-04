import numpy as np
import matplotlib.pyplot as plt

horizon = int(1_500_000/250_000)



#file_path = "DQN_Agent/data/lr_experiments/threshold_False_per_False_001.npy"
#agent_001 = np.load(file_path, allow_pickle=True)

file_path = "DQN_Agent/data/lr_experiments/threshold_False_per_False_00001.npy"
dropout = np.load(file_path, allow_pickle=True)

file_path = "DQN_Agent/data/threshold_False_per_False_no_dropout.npy"
no_drop = np.load(file_path, allow_pickle=True)


plt.figure(1)
plt.xlabel("Round T") 
plt.ylabel("reward in t") 
plt.plot(np.arange(1,horizon+1),dropout, label="With dropout")
plt.plot(np.arange(1,horizon+1),no_drop, label="Without dropout")
plt.grid()
plt.legend()
plt.savefig("./DQN_Agent/images/report/dropout_reduction") # I would have to save fig in order to store it. That is working without depending on .show
plt.show()
print("end")