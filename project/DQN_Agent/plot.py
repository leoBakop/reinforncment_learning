import numpy as np
import matplotlib.pyplot as plt

horizon = int(200_000/10_000)
file_path = "DQN_Agent/data/threshold_False_per_False_model.npy"
# Load the CSV file into a NumPy array
model= np.load(file_path, allow_pickle=True)


file_path = "DQN_Agent/data/threshold_False_per_False_full.npy"
full = np.load(file_path, allow_pickle=True)



plt.figure(1)
plt.xlabel("Round T") 
plt.ylabel("reward in t") 
plt.plot(np.arange(1,horizon+1),model, label="Using a single network")
plt.plot(np.arange(1,horizon+1),full, label="Using two networks")
plt.grid()
plt.legend()
plt.savefig("./DQN_Agent/images/report/model_vs_network") # I would have to save fig in order to store it. That is working without depending on .show
plt.show()
print("end")