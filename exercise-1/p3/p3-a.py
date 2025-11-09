import matplotlib.pyplot as plt
import numpy as np


flexibility = np.linspace(0, 10, 100)

bias_squared = 8 / (flexibility + 1)

variance = 0.15 * flexibility**2

irreducible_error = np.ones_like(flexibility) * 1.5

training_error = 3 * np.exp(-0.4 * flexibility)

test_error = irreducible_error + bias_squared + variance

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each component
plt.plot(flexibility, bias_squared, label="bias squared", linewidth=2)
plt.plot(flexibility, variance, label="variance", linewidth=2)
plt.plot(
    flexibility,
    irreducible_error,
    label="irreducible error",
    linewidth=2,
    color="gray",
)
plt.plot(flexibility, test_error, label="testing error", linewidth=2, color="red")
plt.plot(
    flexibility, training_error, label="training error", linewidth=2, color="blue"
)

# Formatting
plt.xlabel("flexibility", fontsize=12)
plt.ylabel("value", fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.xlim(0, 10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("p3-a.png")
