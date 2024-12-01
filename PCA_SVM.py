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
from scipy.stats import skew, kurtosis

def enhance_image_svd(image, k=75):
    """Enhance an image using Singular Value Decomposition (SVD)."""
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
        'entropy': entropy,
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

def calculate_region_contrast(image, contour):
    """Calculate contrast between tumor region and surrounding area."""
    if contour is None:
        return {'weber_contrast': 0}
    tumor_mask = np.zeros_like(image)
    cv2.drawContours(tumor_mask, [contour], -1, (255), -1)
    kernel = np.ones((50, 50), np.uint8)
    dilated_mask = cv2.dilate(tumor_mask, kernel, iterations=1)
    surrounding_mask = dilated_mask - tumor_mask
    tumor_pixels = image[tumor_mask == 255]
    surrounding_pixels = image[surrounding_mask == 255]
    weber_contrast = (np.mean(tumor_pixels) - np.mean(surrounding_pixels)) / np.mean(surrounding_pixels) if np.mean(surrounding_pixels) != 0 else 0
    return {'weber_contrast': weber_contrast}

def normalize_contrast(image):
    """Apply contrast normalization to the given image."""
    image_normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_normalized

# Data Augmentation Functions
def augment_image(image):
    """Perform data augmentation on the given image."""
    augmented_images = []
    # Original image
    augmented_images.append(image)
    # Rotation
    for angle in [90, 180, 270]:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_180 if angle == 180 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        augmented_images.append(rotated_image)
    # Flipping
    flipped_h = cv2.flip(image, 1)  # Horizontal flip
    flipped_v = cv2.flip(image, 0)  # Vertical flip
    augmented_images.append(flipped_h)
    augmented_images.append(flipped_v)
    # Scaling
    scaled_image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    h, w = image.shape
    scaled_image = scaled_image[(scaled_image.shape[0] - h) // 2: (scaled_image.shape[0] - h) // 2 + h,
                                (scaled_image.shape[1] - w) // 2: (scaled_image.shape[1] - w) // 2 + w]
    augmented_images.append(scaled_image)
    # Adding Gaussian Noise
    noisy_image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)
    augmented_images.append(np.clip(noisy_image, 0, 255))
    noisy_image_high = image + np.random.normal(0, 15, image.shape).astype(np.uint8)
    augmented_images.append(np.clip(noisy_image_high, 0, 255))
    return augmented_images

# Feature Extraction Pipeline
def extract_features(image):
    """Extract features from an image using multiple methods."""
    features = {}
    features.update(calculate_intensity_gradients(image))
    features.update(calculate_glcm_features(image))
    contour = get_contour(image)
    features['circularity'] = calculate_circularity(contour)
    features.update(calculate_region_contrast(image, contour))
    return features

# Load Images, Apply Augmentation, and Extract Features
image_paths = ['11yes.png','12no.png','13no.png','14yes.png','15yes.png','16no.png', '15yes.png', '17yes.png', '18no.png','19yes.png','20yes.png','21no.png','22yes.png']
labels = [1,0,0,1,1,0,1,1,0,1,1,0,1]  # 1 for tumor, 0 for no tumor

feature_list = []
label_list = []
for path, label in zip(image_paths, labels):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    augmented_images = augment_image(image)
    for augmented_image in augmented_images:
        features = extract_features(augmented_image)
        feature_list.append(list(features.values()))
        label_list.append(label)

# Prepare Dataset
X = np.array(feature_list)
y = np.array(label_list)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.9)  # Retain 90% variance
X_pca = pca.fit_transform(X_scaled)
# Output PCA explained variance ratio (feature importance scores)
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=45)

# Classification using SVM
svm_classifier = SVC(kernel='sigmoid', probability=True, random_state=45)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Evaluation
print("SVM Classifier Report:")
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

# Cross Validation
svm_scores = cross_val_score(svm_classifier, X_pca, y, cv=5)
print("\nSVM Cross-Validation Accuracy:", svm_scores.mean())





from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=svm_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import roc_curve, auc

# 计算预测的概率
y_prob_svm = svm_classifier.decision_function(X_test)

# ROC 曲线 (Receiver Operating Characteristic Curve) 和 AUC (Area Under the Curve)
fpr, tpr, _ = roc_curve(y_test, y_prob_svm)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


