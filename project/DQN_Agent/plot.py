import numpy as np
import matplotlib.pyplot as plt

horizon = 1_500_000
evaluate_every = 100_000


file_path = "DQN_Agent/data/final/threshold_False_per_True.npy"
with_per = np.load(file_path, allow_pickle=True)

file_path = "DQN_Agent/data/final/threshold_False_per_False.npy"
without_per = np.load(file_path, allow_pickle=True)



plt.figure(1)
plt.xlabel("Round T") 
plt.ylabel("reward in t")
plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)),with_per, label="Using per") 
plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)),without_per, label="Without per")


plt.grid()
plt.legend()
plt.savefig("./DQN_Agent/images/report/against_random") # I would have to save fig in order to store it. That is working without depending on .show
plt.show()
print("end")