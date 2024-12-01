import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.feature import graycomatrix, graycoprops



# calculate the values of PSNR 计算PSNR值
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

# Step 1: Load and preprocess the image 步骤1：加载并预处理图像
image_path = "9.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image as grayscale

# Check if the image was loaded properly 检查图像是否正确加载
if image is None:
    raise ValueError("Image could not be loaded. Please check the file path.")

matrix = np.array(image)  # Convert to matrix form 转换为矩阵形式

# Ensure the matrix is two-dimensional 确保矩阵是二维的
if len(matrix.shape) != 2:
    raise ValueError("Input matrix must be a two-dimensional array.")



# Step 2: Perform Singular Value Decomposition (SVD) 步骤2：执行奇异值分解（SVD）
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

# Step 3: Reconstruct the image with different numbers of singular values 步骤3：用不同数量的奇异值重建图像
errors = []
psnr_values = []
k_values = range(1, len(S) + 1, 5)  # Use every 5th singular value for reconstruction 使用每5个奇异值进行重建（1,5,10）

for k in k_values:
    # Retain the top k singular values 保留前k个奇异值
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct the image 重建图像
    A_k = np.dot(U_k, np.dot(S_k, Vt_k))
    
    # Calculate reconstruction error 计算重建误差
    error = np.linalg.norm(matrix - A_k, 'fro') / np.linalg.norm(matrix, 'fro')
    errors.append(error)
    
    # Calculate PSNR 计算PSNR
    psnr = calculate_psnr(matrix, A_k)
    psnr_values.append(psnr)

# Plot reconstruction error vs. number of singular values 画出重建误差与奇异值数量的关系
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, errors, marker='o')
plt.xlabel("Number of Singular Values (k)")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error vs. Number of Singular Values")

# Plot PSNR vs. number of singular values 绘制PSNR与奇异值数量的关系图
plt.subplot(1, 2, 2)
plt.plot(k_values[:len(k_values)-1], psnr_values[:len(psnr_values)-1], marker='o', color='r')
plt.xlabel("Number of Singular Values (k)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs. Number of Singular Values")


plt.tight_layout()
plt.show()

# Step 4: Analyze the effect of noise on image enhancement 步骤4：分析噪声对图像增强的影响
noise_levels = [0.01, 0.05, 0.1]

fig1 = plt.figure(figsize=(8, 6))
fig2 = plt.figure(figsize=(15, 6))
plt.figure(fig1.number)
plt.plot(k_values, errors, marker='o', label='Origin',markersize=4)
plt.figure(fig2.number)
plt.plot(k_values[:len(k_values) - 1], psnr_values[:len(psnr_values) - 1],
         marker='o', label='Origin',markersize=4)

for noise_level in noise_levels:
    noisy_image = matrix + noise_level * np.random.normal(loc=0.0, scale=1.0, size=matrix.shape)
    noisy_image = np.clip(noisy_image, 0, 255)

    U_noisy, S_noisy, Vt_noisy = np.linalg.svd(noisy_image, full_matrices=False)

    errors_noisy = []
    psnr_values_noisy = []

    for k in k_values:
        S_k_noisy = np.zeros((k, k))
        np.fill_diagonal(S_k_noisy, S_noisy[:k])
        U_k_noisy = U_noisy[:, :k]
        Vt_k_noisy = Vt_noisy[:k, :]

        A_k_noisy = np.dot(U_k_noisy, np.dot(S_k_noisy, Vt_k_noisy))

        error_noisy = np.linalg.norm(noisy_image - A_k_noisy, 'fro') / np.linalg.norm(noisy_image, 'fro')
        errors_noisy.append(error_noisy)

        psnr_noisy = calculate_psnr(matrix, A_k_noisy)
        psnr_values_noisy.append(psnr_noisy)

    plt.figure(fig1.number)
    plt.plot(k_values, errors_noisy, marker='o', label=f'Noise Level: {noise_level}',markersize=4)

    plt.figure(fig2.number)
    plt.plot(k_values[:len(k_values) - 1], psnr_values_noisy[:len(psnr_values_noisy) - 1],
             marker='o', label=f'Noise Level: {noise_level}', linewidth=1,markersize=4)


plt.figure(fig1.number)
plt.xlabel("Number of Singular Values (k)")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error vs. Number of Singular Values")
plt.legend()

plt.figure(fig2.number)
plt.xlabel("Number of Singular Values (k)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs. Number of Singular Values")
plt.legend()

plt.show()


# Step 5: Analyze computational efficiency 步骤5：分析计算效率
start_time = time.time()
U_full, S_full, Vt_full = np.linalg.svd(matrix, full_matrices=False)
end_time = time.time()
print(f"Full SVD computation time: {end_time - start_time:.4f} seconds")

# Test with reduced number of singular values 减少奇异值数量的测试
k_test = 50
start_time = time.time()
U_k_test = U[:, :k_test]
S_k_test = S[:k_test]
Vt_k_test = Vt[:k_test, :]
A_k_test = np.dot(U_k_test, np.dot(np.diag(S_k_test), Vt_k_test))
end_time = time.time()
print(f"Reduced SVD reconstruction time with k={k_test}: {end_time - start_time:.4f} seconds")
