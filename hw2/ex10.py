from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the function to construct R90
# def construct_R90(n):
#     size = 3*n**2
#     R90 = np.zeros((size, size), dtype=int)
    
#     for i in range(3):
#         for j in range(n**2):
#             col = (j*n) % (n**2-1) 
#             if (j + 1) % n**2 == 0 and j >= n**2-1:
#                 col += n**2-1
#             R90[i*n**2 + j, i*n**2 + col] = 1
                
#     return R90

def create_n_permutation_matrix(n):
    size = n ** 2
    j = np.arange(size)
    col = (j * n) % (size - 1)
    col += np.where((j + 1) % size == 0, size - 1, 0)
    m = np.zeros((size, size), dtype=int)
    m[j, col] = 1
    return m

def construct_R90(n):
    size = 3 * n ** 2
    R90 = np.zeros((size, size), dtype=int)
    
    # Populate the R90 matrix
    for k in range(3):
        m = create_n_permutation_matrix(n)
        R90[k * n ** 2 : (k + 1) * n ** 2, k * n ** 2 : (k + 1) * n ** 2] = m
    
    return R90


# Load the image
image = Image.open("imgs/dog.jpeg")
image = image.resize((92, 92))

plt.imshow(image)
plt.savefig("imgs/dog_resized.jpeg")
plt.show()

# Convert the image to numpy array
image_array = np.array(image)

# Get the dimensions of the image
height, width, channels = image_array.shape
print(f"Image shape: {image_array.shape}")

# Convert the image to column-stack representation
column_stack_representation = image_array.transpose(2,0,1).reshape(-1)
print(f"Column-stack representation shape: {column_stack_representation.shape}")

n = height  # Assuming square image
R90 = construct_R90(n)
R90_inv = np.linalg.inv(R90)
print(f"R90 shape: {R90.shape}")
print(f"R90_inv shape : ", R90_inv.shape)

rotated_column_stack = np.dot(R90, column_stack_representation[:,None]).astype(np.uint32).squeeze()

# Reshape the rotated column-stack representation back to image shape
rotated_image = rotated_column_stack.reshape(channels, height, width).transpose(1,2,0)

plt.imshow(rotated_image)
plt.savefig("imgs/dog_rotated.jpeg")
plt.show()
