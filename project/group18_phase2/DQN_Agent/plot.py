import numpy as np
import matplotlib.pyplot as plt

horizon = 3_000_000
evaluate_every = 100_000


file_path = "data/final/against_tight/threshold_True_per_False_loose_False_best_False.npy"
without_per = np.load(file_path, allow_pickle=True)

file_path = "data/final/against_tight/threshold_True_per_True_loose_False_best_False.npy"
with_per = np.load(file_path, allow_pickle=True)



plt.figure(1)
plt.xlabel("Round T") 
plt.ylabel("reward in t")
plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)),with_per, label="Using per") 
plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)),without_per, label="Without per")
plt.grid()
plt.legend()
plt.savefig("./images/report/against_tight_threshold") # I would have to save fig in order to store it. That is working without depending on .show
plt.show()
print("end")