import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'apr24/Sample_Shoulder_Conds.jpg'
img = Image.open(image_path)

# Convert to numpy array
img_array = np.array(img)

# Print basic information
print(f"Image shape: {img_array.shape}")
print(f"Image data type: {img_array.dtype}")
print(f"Min value: {img_array.min()}")
print(f"Max value: {img_array.max()}")
print(f"Mean value: {img_array.mean()}")

# Print a small sample of pixels
print("\nSample pixel values (top-left corner):")
print(img_array[0:5, 0:5])

# Show histogram of pixel values
plt.figure(figsize=(10, 6))
if len(img_array.shape) == 3:  # Color image
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.hist(img_array[:,:,i].ravel(), bins=256, color=color, alpha=0.5)
else:  # Grayscale image
    plt.hist(img_array.ravel(), bins=256, color='gray')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Image Histogram')
plt.savefig('histogram.png')  # Save histogram to view later
print("\nHistogram saved as 'histogram.png'")