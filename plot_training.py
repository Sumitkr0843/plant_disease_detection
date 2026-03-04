import matplotlib.pyplot as plt

epochs = list(range(1,11))

train_loss = [
992.4602, 673.5113, 645.9047, 629.2264, 619.2590,
609.9656, 607.2709, 598.7700, 594.6848, 586.8323
]

val_loss = [
140.3854, 130.5562, 131.1339, 126.9742, 126.2951,
125.1066, 125.2240, 124.3901, 124.0437, 123.6357
]

train_acc = [
84.99, 94.58, 95.47, 96.28, 96.61,
97.06, 97.13, 97.43, 97.66, 98.00
]

val_acc = [
96.66, 98.52, 98.15, 99.14, 99.34,
99.41, 99.43, 99.51, 99.60, 99.65
]

plt.figure(figsize=(8,8))

# LOSS GRAPH
plt.subplot(2,1,1)
plt.plot(epochs, train_loss, label="Training loss", color="blue")
plt.plot(epochs, val_loss, label="Validation loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# ACCURACY GRAPH
plt.subplot(2,1,2)
plt.plot(epochs, train_acc, '--', label="Training accuracy")
plt.plot(epochs, val_acc, '--', label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("training_results.png", dpi=300)

plt.show()