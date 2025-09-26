import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the saved plot images
img1 = mpimg.imread("homophily_f_deterministic_experiment_mean.png")
img2 = mpimg.imread("homophily_f_deterministic_experiment_std.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.imshow(img1)
ax1.axis("off")  # hide axes for image
ax2.imshow(img2)
ax2.axis("off")

fig.suptitle("Combined Imported Plots", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

