from matplotlib import pyplot as plt

RATIONALITY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
exp_utility = [0.54, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79, 0.83, 0.87]
og_utility = [0.53, 0.57, 0.60, 0.63, 0.66, 0.69, 0.72, 0.75, 0.79]

exp_acc = [0.88, 0.88, 0.88, 0.88, 0.87, 0.87, 0.87, 0.87, 0.87]
og_acc = [0.90, 0.90, 0.90,  0.90, 0.90, 0.90, 0.90, 0.90, 0.90]


plt.xlabel("Human Rationality Probability")
plt.ylabel("Utility")
plt.plot(RATIONALITY, exp_utility, label="When Optimized for Expected Utility")
plt.plot(RATIONALITY, og_utility, label="Regular loss")
leg = plt.legend(loc='upper center')
plt.savefig("results/mlp_moons_ut" + ".png")
plt.close()



plt.xlabel("Human Rationality Probability")
plt.ylabel("Accuracy")
plt.plot(RATIONALITY, exp_acc, label="When Optimized for Expected Utility")
plt.plot(RATIONALITY, og_acc, label="Regular loss")
leg = plt.legend(loc='upper center')
plt.savefig("results/mlp_moons_acc" + ".png")
plt.close()


STD = [1e-3, 1e-2, 1e-1, 0.2, 0.5]

exp_utility = [0.92, 0.92, 0.92, 0.87, 0.77]
og_utility = [0.83, 0.83, 0.83, 0.77, 0.69]

exp_acc = [0.88, 0.87, 0.87, 0.87, 0.87]
og_acc = [0.90, 0.90, 0.90,  0.90, 0.90]

plt.xlabel("Standard Deviation")
plt.ylabel("Utility")
plt.plot(STD, exp_utility, label="When Optimized for Expected Utility")
plt.plot(STD, og_utility, label="Regular loss")
leg = plt.legend(loc='upper center')
plt.savefig("results/mlp_moons_ut_sd" + ".png")
plt.close()



plt.xlabel("Standard Deviation")
plt.ylabel("Accuracy")
plt.plot(STD, exp_acc, label="When Optimized for Expected Utility")
plt.plot(STD, og_acc, label="Regular loss")
leg = plt.legend(loc='upper center')
plt.savefig("results/mlp_moons_acc_sd" + ".png")
plt.close()
