import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from scipy import stats


# Feature Extraction Functions
def calculate_intensity_gradients(image):
    """Calculate intensity gradient features."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    direction = np.arctan2(sobely, sobelx)
    gradient_features = {
        'mean_magnitude': np.mean(magnitude),
        'std_magnitude': np.std(magnitude),
        'max_magnitude': np.max(magnitude),
        'mean_direction': np.mean(direction),
        'direction_entropy': stats.entropy(np.histogram(direction, bins=36)[0])
    }
    return gradient_features

def calculate_glcm_features(image):
    """Calculate GLCM features for the given image."""
    if image.dtype != np.uint8:
        image = (image / image.max() * 255).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)
    entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
    glcm_features = {
        'contrast': np.mean(graycoprops(glcm, 'contrast')),
        'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'energy': np.mean(graycoprops(glcm, 'energy')),
        'correlation': np.mean(graycoprops(glcm, 'correlation')),
        'ASM': np.mean(graycoprops(glcm, 'ASM')),
        'entropy': entropy
    }
    return glcm_features

def get_contour(image, threshold_method='otsu'):
    """Get the largest contour from the image."""
    if image.dtype != np.uint8:
        image = (image / image.max() * 255).astype(np.uint8)
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        return max(contours, key=cv2.contourArea)
    return None

def calculate_circularity(contour):
    """Calculate circularity of the detected contour."""
    if contour is None:
        return 0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity


def calculate_region_contrast(image, contour,method):
    """Calculate contrast between tumor region and surrounding area."""
    if contour is None:
        return {'michelson_contrast': 0}

    # Create mask
    tumor_mask = np.zeros_like(image)
    cv2.drawContours(tumor_mask, [contour], -1, (255), -1)
    kernel = np.ones((50, 50), np.uint8)
    dilated_mask = cv2.dilate(tumor_mask, kernel, iterations=1)
    surrounding_mask = dilated_mask - tumor_mask

    tumor_pixels = image[tumor_mask == 255].astype(np.float64)
    surrounding_pixels = image[surrounding_mask == 255].astype(np.float64)



    if len(tumor_pixels) > 0 and len(surrounding_pixels) > 0:

        # Calculate Michelson Contrast
        if method == 'mich':
            max_tumor = np.max(tumor_pixels)
            min_surrounding = np.min(surrounding_pixels)
            denominator_m = max_tumor + min_surrounding
            michelson_contrast = (max_tumor - min_surrounding) / denominator_m if denominator_m != 0 else 0

            return {'michelson_contrast': float(michelson_contrast)}

        # Calculate Weber Contrast
        if method == 'weber':
            weber_contrast = (np.mean(tumor_pixels) - np.mean(surrounding_pixels)) /np.mean(surrounding_pixels) if np.mean(surrounding_pixels) != 0 else 0
            return {'weber_contrast': weber_contrast}

    else:
        return None

def normalize_contrast(image):
    """Apply contrast normalization to the given image."""
    image_normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_normalized


def augment_image(image):
    """Perform comprehensive data augmentation on the given image and visualize results."""
    augmented_images = []
    titles = []

    # Original image
    augmented_images.append(image)
    titles.append('Original [1]')

    # Rotation
    angles = [90, 180, 270]
    for angle in angles:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle == 90
        else cv2.ROTATE_180 if angle == 180
        else cv2.ROTATE_90_COUNTERCLOCKWISE)
        augmented_images.append(rotated_image)
        titles.append(f'Rotated {angle}° [{angles.index(angle)+2}]')

    # Flipping
    flipped_h = cv2.flip(image, 1)  # Horizontal flip
    flipped_v = cv2.flip(image, 0)  # Vertical flip
    augmented_images.extend([flipped_h, flipped_v])
    titles.extend(['Horizontal Flip [5]', 'Vertical Flip [6]'])


    # Adding Gaussian Noise
    noisy_image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy_image = np.clip(noisy_image, 0, 255)
    augmented_images.append(noisy_image)
    titles.append('Gaussian Noise [7]')

    # Scaling
    scaled_image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    h, w = image.shape
    scaled_image = scaled_image[(scaled_image.shape[0] - h) // 2: (scaled_image.shape[0] - h) // 2 + h,
                   (scaled_image.shape[1] - w) // 2: (scaled_image.shape[1] - w) // 2 + w]
    augmented_images.append(scaled_image)
    titles.append('Scaled (1.2x) [8]')

    # Create subplot grid
    num_images = len(augmented_images)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(15, 3 * num_rows))

    # Plot each image
    for idx, (img, title) in enumerate(zip(augmented_images, titles)):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return augmented_images

# Feature Extraction Pipeline
def extract_features(image, method):
    """Extract features from an image using multiple methods."""
    features = {}
    features.update(calculate_intensity_gradients(image))
    features.update(calculate_glcm_features(image))
    contour = get_contour(image)
    features['circularity'] = calculate_circularity(contour)
    features.update(calculate_region_contrast(image, contour, method))
    return features



con = {} # store the contrast value
image_paths = ['15yes.png']
labels = [0]

for method in ['mich', 'weber']:
    feature_list = []
    label_list = []

    for path, label in zip(image_paths, labels):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).copy()
        image = cv2.resize(image, (256, 256))

        augmented_images = augment_image(image)
        for augmented_image in augmented_images:
            features = extract_features(augmented_image, method)  # 传入method参数
            feature_list.append(list(features.values()))
            label_list.append(label)

    # Prepare Dataset
    X = np.array(feature_list)
    y = np.array(label_list)

    # Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print(method)
    # print(X_scaled[:, 13])

    con[method] = X_scaled[:, 13]


# Visualization
plt.figure(figsize=(12, 6))

# get data for the dict
mich_values = con.get('mich', [])
weber_values = con.get('weber', [])


x_labels = [f'image{i+1}' for i in range(len(weber_values))]  # 使用weber值的长度
x = np.arange(len(x_labels))


if len(mich_values) > 0:
    plt.plot(x, mich_values, marker='o', linestyle='-',label='Michelson Contrast', linewidth=2)
if len(weber_values) > 0:
    plt.plot(x, weber_values, marker='o', linestyle='--',label='Weber Contrast', linewidth=2)


plt.xlabel('Image Number')
plt.ylabel('Contrast Value')
plt.title('Comparison of Michelson and Weber Contrast')
plt.xticks(x, x_labels, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# print("Michelson values:", mich_values)
# print("Weber values:", weber_values)
