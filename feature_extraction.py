import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import time



def normalize_contrast(image):
    """Apply contrast normalization to the given image."""
    image_normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_normalized

def enhance_image_svd(image, k=75):
    """Enhance an image using Singular Value Decomposition (SVD), based on the result we get previously."""
    matrix = np.array(image)
    if len(matrix.shape) != 2:
        raise ValueError("Input matrix must be a two-dimensional array.")
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    A_k = np.dot(U_k, np.dot(S_k, Vt_k))
    A_k = np.clip(A_k, 0, 255).astype(np.uint8)
    return A_k

def calculate_intensity_gradients(image,show:bool):
    """Calculate intensity gradient features."""

    # use sobel operator to calculate gradient in vertical and horizontal direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    direction = np.arctan2(sobely, sobelx)

    # calculate some statistics
    gradient_features = {
        'mean_magnitude': np.mean(magnitude),
        'std_magnitude': np.std(magnitude),
        'max_magnitude': np.max(magnitude),
        'mean_direction': np.mean(direction),
        'direction_entropy': stats.entropy(np.histogram(direction, bins=36)[0])
    }

    # visualization
    plt.figure()
    plt.imshow(magnitude, cmap='jet')
    plt.title('Gradient Magnitude')
    plt.colorbar()
    plt.savefig('Gradient Magnitude2')

    if show:
        plt.show()

    return gradient_features

def calculate_glcm_features(image):
    """Calculate GLCM features for the given image."""

    if image.dtype != np.uint8:
        image = (image / image.max() * 255).astype(np.uint8)

    # get the glcm matrix
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

    # normalize the matrix
    glcm_norm = glcm / np.sum(glcm)
    glcm_norm = glcm_norm + np.finfo(float).eps

    glcm_features = {
        'contrast': np.mean(graycoprops(glcm, 'contrast')),
        'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'energy': np.mean(graycoprops(glcm, 'energy')),
        'correlation': np.mean(graycoprops(glcm, 'correlation')),
        'ASM': np.mean(graycoprops(glcm, 'ASM')),
        'Entropy': -np.sum(glcm_norm * np.log2(glcm_norm))
    }

    return glcm_features

def get_contour(image, threshold_method='otsu'):
    """Get the largest contour from the image."""

    if image.dtype != np.uint8:
        image = (image / image.max() * 255).astype(np.uint8)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # denoising
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # chose the largest contour
    if len(contours) > 0:
        contour_max = max(contours, key=cv2.contourArea)
        return contour_max
    return None


def calculate_circularity(image,contour, show:bool):
    """Calculate circularity of the detected contour."""

    if contour is None:
        return 0

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # Visualization
    if show:
        plt.figure()
        image_max_contour = cv2.drawContours(image.copy(), [contour], -1, (0, 255, 0), 2)
        plt.imshow(image_max_contour, cmap='gray')
        plt.title(f"Circularity: {circularity:.4f}")
        plt.savefig('Circularity2')
        plt.show()

    return circularity


def calculate_region_contrast(image, contour, show:bool):
    """Calculate contrast between tumor region and surrounding area."""

    if contour is None:
        return None

    # create tumor mask
    tumor_mask = np.zeros_like(image)
    cv2.drawContours(tumor_mask, [contour], -1, (255), -1)

    # expand tumor mask
    kernel = np.ones((50, 50), np.uint8)
    dilated_mask = cv2.dilate(tumor_mask, kernel, iterations=1)

    # created surrounding tissues mask
    surrounding_mask = dilated_mask - tumor_mask

    # Get the pixel value of the corresponding area
    tumor_pixels = image[tumor_mask == 255]
    surrounding_pixels = image[surrounding_mask == 255]

    contrast_features = {
        'tumor_mean': np.mean(tumor_pixels),
        'tumor_std': np.std(tumor_pixels),
        'surrounding_mean': np.mean(surrounding_pixels),
        'surrounding_std': np.std(surrounding_pixels),
        'weber_contrast': (np.mean(tumor_pixels) - np.mean(surrounding_pixels)) /
                          np.mean(surrounding_pixels) if np.mean(surrounding_pixels) != 0 else 0,
    }

    #visualization
    if show:
        plt.figure(figsize=(15, 5))

        # original image and contour
        plt.subplot(131)
        overlay = image.copy()
        overlay = cv2.drawContours(overlay, [contour], -1, 255, 2)
        plt.imshow(overlay, cmap='gray')
        plt.title('Original with Contour')

        # tumor mask
        plt.subplot(132)
        plt.imshow(tumor_mask, cmap='gray')
        plt.title('Tumor Mask')

        # surrounding mask
        plt.subplot(133)
        plt.imshow(surrounding_mask, cmap='gray')
        plt.title('Surrounding Mask')
        plt.savefig('calculating process2')
        plt.show()

        # Histogram
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.hist(tumor_pixels, bins=50, alpha=0.5, label='Tumor', density=True)
        plt.hist(surrounding_pixels, bins=50, alpha=0.5, label='Surrounding', density=True)
        plt.legend()
        plt.title('Grayscale Distribution')

        # boxplot
        plt.subplot(122)
        plt.boxplot([tumor_pixels, surrounding_pixels],
                    labels=['Tumor', 'Surrounding'])
        plt.title('Grayscale Box Plot')

        plt.tight_layout()
        plt.savefig('Grayscale Distribution2')
        plt.show()

    return contrast_features



#############################################################################
# Load image as grayscale
path1 = '70_big.png'
image1 = cv2 . imread (path1, cv2 . IMREAD_GRAYSCALE ).copy()

# Unify image sizes
image1 = cv2.resize(image1, (256, 256))

# image preprocessing
image1 = enhance_image_svd(image1, k=75)
image1 = normalize_contrast(image1)


G1 = calculate_intensity_gradients(image1,show=True)
print("\n=== intensity gradient ===")
print(G1)

glcm1 = calculate_glcm_features(image1)
print("\n=== GLCM ===")
print(glcm1)

contour1 = get_contour(image1)

calculate_circularity(image1,contour1, show=True)

contrast1 = calculate_region_contrast(image1, contour1, show=True)
print("\n=== contrast ===")
print(contrast1)


## PCA
def extract_features(image):
    # Calculate all features and combine
    contour = get_contour(image)
    features = {}
    features.update(calculate_intensity_gradients(image, show=False))
    features.update(calculate_glcm_features(image))
    features['circularity'] = calculate_circularity(image, contour, show=False)
    features.update(calculate_region_contrast(image, contour, show=False))
    return features

## load the image
path1 = '22yes.png'
path2 = '21no.png'
path3 = '17yes.png'
path4 = '18no.png'
path5 = '19yes.png'
path6 = '16no.png'
image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(path4, cv2.IMREAD_GRAYSCALE)
image5 = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)
image6 = cv2.imread(path6, cv2.IMREAD_GRAYSCALE)

# extract features
features1 = extract_features(image1)
features2 = extract_features(image2)
features3 = extract_features(image3)
features4 = extract_features(image4)
features5 = extract_features(image5)
features6 = extract_features(image6)


# Merge all features into one feature matrix
features_list = [features1, features2,features3,features4,features5,features6]  # 假设有更多样本时逐个提取
feature_matrix = np.array([list(f.values()) for f in features_list])
labels = np.array([1, 0,1,0,1,0])  #  1 stand for there is tumor, 0 is no tumor

# 1. use PCA to reduce dimensions
scaler = StandardScaler()  # Standardized features
feature_matrix_scaled = scaler.fit_transform(feature_matrix)  # Scaling Features
pca = PCA(n_components=0.9)  # Keep 90% variance
features_pca = pca.fit_transform(feature_matrix_scaled)
print("Feature dimensions after PCA：", features_pca.shape)

# 2. calculate feature important score
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features_pca, labels)
feature_importances = model.feature_importances_
print("Feature important score：", feature_importances)


pca = PCA(n_components=2)  # Select 2 principal components for visualization
features_pca_2d = pca.fit_transform(feature_matrix_scaled)

# Visualizing 2D PCA results
plt.figure(figsize=(8, 6))
plt.scatter(features_pca_2d[:, 0], features_pca_2d[:, 1], c=labels, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Result (2D)")
plt.colorbar(label='Label')
plt.show()


# 3. Cross-validation using different feature sets
# can choose different feature sets to observe the effect, such as the full feature set, features after PCA dimensionality reduction, etc.
# using larger datasets in real-world applications


scores = cross_val_score(model, features_pca, labels, cv=3)  # 5-fold cross validation
print("Cross-validation score of PCA features：", scores.mean())


# 4. Testing feature stability under different image conditions
# After adding different noises, extract features and calculate mean and variance
noise_levels = [0.01, 0.05, 0.1]
for noise_level in noise_levels:
    noisy_image = image1 + noise_level * np.random.normal(size=image1.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_features = extract_features(noisy_image)
    print(f"Feature mean at noise level {noise_level} ：", np.mean(list(noisy_features.values())))
    print(f"Feature stander deviation at noise level {noise_level} ：", np.std(list(noisy_features.values())))

# 5. Calculate feature computation efficiency
start_time = time.time()
features = extract_features(image1)
end_time = time.time()
print("Time of feature extraction：", end_time - start_time, "second")



