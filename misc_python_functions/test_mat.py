import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
data = np.random.randn(1000) * 10 + 50  # Normally distributed around 50

# Use qcut to determine bin edges
_, bin_edges = pd.qcut(data, q=4, retbins=True)  # `retbins=True` gives bin edges

# Create the plot
plt.figure(figsize=(10, 6))

# Plot histogram using bin_edges
plt.hist(data, bins=bin_edges, color="skyblue", edgecolor="black", alpha=0.7, histtype="barstacked", label="Data Distribution")

# Customize plot
plt.xlabel("Data Value")
plt.ylabel("Frequency")
plt.title("Histogram Using qcut Bin Edges")
plt.xticks(bin_edges.round(2))  # Show bin edges as x-ticks
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()

# Show the plot
plt.show()
