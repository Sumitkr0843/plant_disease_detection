import matplotlib.pyplot as plt

epochs = list(range(1,11))

train_loss = [
1001.6641, 676.9747, 647.5307, 626.9533, 618.4579,
614.4521, 601.5680, 597.1573, 590.5519, 588.8063
]

val_loss = [
133.4555, 133.0077, 128.0070, 128.1981, 126.5473,
125.7491, 123.5922, 124.3610, 123.3993, 123.7033
]

train_acc = [
84.30, 94.36, 95.48, 96.32, 96.61,
96.77, 97.37, 97.56, 97.82, 97.90
]

val_acc = [
98.14, 97.93, 98.86, 98.81, 99.10,
99.30, 99.71, 99.50, 99.65, 99.71
]

plt.figure(figsize=(8,8))

# LOSS GRAPH
plt.subplot(2,1,1)
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# ACCURACY GRAPH
plt.subplot(2,1,2)
plt.plot(epochs, train_acc, '--', label="Training Accuracy")
plt.plot(epochs, val_acc, '--', label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("training_results.png", dpi=300)

plt.show()