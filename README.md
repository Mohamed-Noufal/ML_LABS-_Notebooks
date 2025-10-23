# ML - LABS - Notebooks
 
A comprehensive collection of machine learning notebooks covering fundamental concepts to advanced applications. This repository contains hands-on implementations, detailed explanations, and real-world examples across supervised learning, unsupervised learning, NLP, computer vision, and Hugging Face transformers.
 
## 🎯 Project Overview

This repository serves as both a learning resource and a reference implementation for various machine learning algorithms and techniques. Each notebook is carefully structured to explain concepts, implement algorithms from scratch, and demonstrate practical applications with clear file paths and usage examples.

## 📚 Detailed Repository Structure

### 1. Supervised Learning (`Supervised machine learning labs/`)
Path: `Supervised machine learning labs/`

#### Classical Machine Learning
- **Logistic Regression** (`C1_W3_Logistic_Regression.ipynb`)
  - **File Path**: `Supervised machine learning labs/C1_W3_Logistic_Regression.ipynb`
  - **Algorithm**: Binary classification using sigmoid function
  - **Example Usage**: Email spam detection
    ```python
    # Example: Medical diagnosis
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)  # X: patient features, y: disease status
    prediction = model.predict(new_patient_data)
    ```
  - **Key Concepts**: Sigmoid function, decision boundaries, regularization
  - **Real-world Application**: Medical diagnosis, credit risk assessment

- **Gradient Descent Deep Dive** (`C1_W3_Lab06_Gradient_Descent_Soln.ipynb`)
  - **File Path**: `Supervised machine learning labs/C1_W3_Lab06_Gradient_Descent_Soln.ipynb`
  - **Algorithm**: Parameter optimization through iterative updates
  - **Example Usage**: Neural network training
    ```python
    # Example: Linear regression weight update
    w = w - learning_rate * dw  # dw: gradient of loss w.r.t. weights
    b = b - learning_rate * db  # db: gradient of loss w.r.t. bias
    ```
  - **Key Concepts**: Learning rate tuning, convergence analysis
  - **Application**: Model weight optimization

- **Feature Scaling and Learning Rate** (`C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln-Copy1.ipynb`)
  - **File Path**: `Supervised machine learning labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln-Copy1.ipynb`
  - **Algorithm**: Feature normalization (Standardization/MinMax scaling)
  - **Example Usage**: Neural network preprocessing
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Features with mean=0, std=1
    ```
  - **Impact**: Faster convergence, better model performance

- **Scikit-learn Practical Guide** (`C1_W3_Lab07_Scikit_Learn_Soln.ipynb`)
  - **File Path**: `Supervised machine learning labs/C1_W3_Lab07_Scikit_Learn_Soln.ipynb`
  - **Framework**: Industry-standard ML library
  - **Example Usage**: Pipeline building
    ```python
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    ```

#### Additional Labs
- **Cost Function** (`C1_W3_Lab05_Cost_Function_Soln.ipynb`)
  - Path: `Supervised machine learning labs/C1_W3_Lab05_Cost_Function_Soln.ipynb`
- **Multi-class Assignment** (`C2W4_Assignment.ipynb`)
  - Path: `Supervised machine learning labs/C2W4_Assignment.ipynb`

### 2. Unsupervised Learning (`Unsupervised machine learning labs/`)
Path: `Unsupervised machine learning labs/`

- **PCA Visualization** (`C3_W2_Lab01_PCA_Visualization_Examples.ipynb`)
  - **File Path**: `Unsupervised machine learning labs/C3_W2_Lab01_PCA_Visualization_Examples.ipynb`
  - **Algorithm**: Dimensionality reduction via principal components
  - **Example Usage**: Data visualization and compression
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_high_dim)  # Reduce to 2D for plotting
    ```
  - **Application**: Feature extraction, data compression

- **K-Means Clustering** (`C3_W1_KMeans_Assignment.ipynb`)
  - **File Path**: `Unsupervised machine learning labs/C3_W1_KMeans_Assignment.ipynb`
  - **Algorithm**: Centroid-based clustering
  - **Example Usage**: Customer segmentation
    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)
    customer_segments = kmeans.fit_predict(customer_data)
    ```

- **Anomaly Detection** (`C3_W1_Anomaly_Detection.ipynb`)
  - **File Path**: `Unsupervised machine learning labs/C3_W1_Anomaly_Detection.ipynb`
  - **Algorithm**: Statistical outlier detection using Gaussian models
  - **Example Usage**: Network intrusion detection
    ```python
    # Calculate anomaly score based on probability density
    p = multivariate_gaussian(X, mu, cov)  # mu: mean, cov: covariance
    anomalies = np.where(p < epsilon)  # epsilon: threshold
    ```
  - **Application**: Fraud detection, manufacturing quality control

- **Recommendation Systems** (`C3_W2_Collaborative_RecSys_Assignment.ipynb`)
  - **File Path**: `Unsupervised machine learning labs/C3_W2_Collaborative_RecSys_Assignment.ipynb`
  - **Algorithm**: Collaborative filtering (user-item matrix)
  - **Example Usage**: Movie recommendations
    ```python
    # Matrix factorization for user-item ratings
    U, V = matrix_factorization(R)  # R: Ratings matrix, U: user features, V: item features
    predicted_rating = np.dot(U[user_id], V[item_id])
    ```
  - **Application**: Content personalization, e-commerce recommendations

### 3. Natural Language Processing (`NLP -N/`)
Path: `NLP -N/`

- **Time Series with Neural Networks**
  - **Simple RNN** (`Simple RNN.ipynb`) - Path: `NLP -N/Simple RNN.ipynb`
  - **LSTM Networks**: Single Layer (`W3_Lab_1_single_layer_LSTM.ipynb`), Multi Layer (`W3_Lab_2_multiple_layer_LSTM.ipynb`)
  - **Convolutional LSTM** (`Convolutions with LSTM.ipynb`)
  - **Bi-directional LSTM** (`W3_Lab_5_sarcasm_with_bi_LSTM.ipynb`)
  - **1D Convolutional** (`W3_Lab_6_sarcasm_with_1D_convolutional.ipynb`)

- **Text Generation**
  - **Shakespeare Text Generation** (`Generating Text with Neural Networks/C3_W4_Lab_3_text_generation.ipynb`)
    - **Data**: `NLP -N/Generating Text with Neural Networks/data/shakespeare.txt`
    - **Algorithm**: Character-level RNN text generation
    - **Example**: Generate new Shakespeare-like text
  - **Irish Lyrics Generation** (`Generating Text from Irish Lyrics/C3_W4_Lab_2_irish_lyrics.ipynb`)
    - **Data**: `NLP -N/Generating Text from Irish Lyrics/data/irish-lyrics-eof.txt`

- **Predicting Next Word** (`perdection the next word/C3W4_Assignment.ipynb`)
  - **Algorithm**: Language model for text prediction
  - **Data**: `NLP -N/perdection the next word/data/sonnets.txt`
  - **Example**: Auto-complete and text suggestions

- **Sunspots Prediction**
  - **DNN** (`C4_W4_Lab_2_Sunspots_DNN.ipynb`)
  - **CNN+RNN+DNN** (`C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb`)

### 4. Computer Vision and Transfer Learning (`Supervised machine learning labs/Image_data_preprocessing/`, `TF/`)
- **Image Preprocessing** (`Supervised machine learning labs/Image_data_preprocessing/`)
  - **Labs**: `lab_1/`, `lab_2/`, `lab_3/`
  - **Datasets**: `horse-or-human/`, `validation-horse-or-human/`

- **Transfer Learning** (`TF/C2_W3_Lab_1_transfer_learning.ipynb`)
  - **File Path**: `TF/C2_W3_Lab_1_transfer_learning.ipynb`
  - **Algorithm**: Fine-tuning pre-trained models
  - **Example Usage**: Custom image classification with MobileNetV2
    ```python
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    ```

### 5. Hugging Face Transformers (`Hugging face -object detection/`)
Path: `Hugging face -object detection/`

- **Object Detection** (`L8_object_detection.ipynb`)
  - **Algorithm**: DETR (Detection Transformer)
  - **Example**: Detect objects in images with bounding boxes
    ```python
    from transformers import DetrImageProcessor, DetrForObjectDetection
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    ```

- **Image Segmentation** (`L9_segmentation.ipynb`)
  - **Algorithm**: Mask R-CNN
  - **Example**: Instance segmentation for precise object boundaries

- **Image Retrieval** (`L10_image_retrieval.ipynb`)
  - **Algorithm**: CLIP (Contrastive Language-Image Pretraining)
  - **Example**: Search images using text queries
    ```python
    import clip
    model, preprocess = clip.load("ViT-B/32")
    text_features = model.encode_text(clip.tokenize(["a red apple"]))
    image_features = model.encode_image(preprocessed_image)
    similarity = text_features @ image_features.T
    ```

- **Image Captioning** (`L11_image_captioning.ipynb`)
  - **Algorithm**: Vision-Encoder-Decoder with Transformer
  - **Example**: Generate descriptive captions for images

- **Visual Question Answering** (`L12_visual_q_and_a.ipynb`)
  - **Algorithm**: Multi-modal transformers
  - **Example**: Answer questions about image content
    ```python
    from transformers import ViltProcessor, ViltForQuestionAnswering
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # Question: "What color is the car?"
    ```

- **Zero-Shot Classification** (`L13_Zero_Shot_Image_Classification.ipynb`)
  - **Algorithm**: CLIP zero-shot classification
  - **Example**: Classify images without specific training
    ```python
    # Classify cat vs dog without training data
    text_inputs = clip.tokenize(["a photo of a cat", "a photo of a dog"])
    text_features = model.encode_text(text_inputs)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    ```

- **Model Deployment** (`L14_deployment.ipynb`)
  - **Implementation**: Flask/FastAPI web service deployment

### 6. TensorFlow Fundamentals (`TF/`)
Path: `TF/`

- **Beyond Hello World** (`C1_W2_Lab_1_beyond_hello_world.ipynb`)
- **TensorFlow Assignment** (`C1W3_Assignment.ipynb`)
- **Cat vs Dog Classification** (`C2W2_cat&dog.ipynb`)

### 7. Standalone Notebooks (Root Level)

- **Feature Engineering** (`_FeatEng_PolyReg_Soln.ipynb`) - Polynomial regression preprocessing
- **Decision Boundary** (`Decision_Boundary_Soln.ipynb`) - Visualization of classification boundaries
- **Sigmoid Function** (`Sigmoid_function_Soln.ipynb`) - Activation function deep dive
- **Sklearn Gradient Descent** (`Sklearn_GD_Soln.ipynb`) - Optimized gradient descent implementations

## 🔧 Algorithms and Real-World Applications

### Supervised Learning Algorithms with Examples

#### Logistic Regression (`Supervised machine learning labs/C1_W3_Logistic_Regression.ipynb`)
**Algorithm Type**: Binary Classification
**Key Concept**: Sigmoid activation function converts linear output to probability

**Real-World Examples:**
1. **Medical Diagnosis**:
   ```python
   # Predict disease probability based on patient vitals
   model = LogisticRegression()
   model.fit(patients[['age', 'blood_pressure', 'cholesterol']], patients['has_disease'])
   new_patient_prob = model.predict_proba([[45, 140, 200]])[0][1]  # Probability of disease
   ```

2. **Email Spam Classification**:
   ```python
   # Classify emails as spam/not spam
   features = ['word_freq_make', 'word_freq_address', 'capital_run_length_average']
   model.fit(X_train[features], y_train)
   prediction = model.predict_proba(email_features)  # Spam probability
   ```

#### Gradient Descent (`Supervised machine learning labs/C1_W3_Lab06_Gradient_Descent_Soln.ipynb`)
**Algorithm**: Optimization through parameter updates
**Example**: Training neural network weights
```python
# Manual gradient descent implementation
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    W = np.zeros(n)
    b = 0

    for _ in range(epochs):
        # Forward pass
        y_pred = X @ W + b
        loss = np.mean((y_pred - y) ** 2)

        # Calculate gradients
        dW = (2/m) * X.T @ (y_pred - y)
        db = (2/m) * np.sum(y_pred - y)

        # Update parameters
        W -= learning_rate * dW
        b -= learning_rate * db

    return W, b
```

#### Feature Scaling (`Supervised machine learning labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln-Copy1.ipynb`)
**Purpose**: Normalize features for better convergence
```python
# Standardization vs Normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (preferred for gradient descent)
scaler_std = StandardScaler()  # Result: mean=0, std=1
X_standardized = scaler_std.fit_transform(X)

# Min-Max scaling
scaler_norm = MinMaxScaler()  # Result: [0,1] range
X_normalized = scaler_norm.fit_transform(X)
```

### Unsupervised Learning Algorithms with Examples

#### PCA (`Unsupervised machine learning labs/C3_W2_Lab01_PCA_Visualization_Examples.ipynb`)
**Algorithm**: Dimensionality reduction through eigenvalue decomposition

**Practical Examples:**
1. **Image Compression**:
   ```python
   # Reduce MNIST image dimensions from 784 to 50
   pca = PCA(n_components=50)
   X_compressed = pca.fit_transform(X_images)  # Keep 95% of variance
   X_reconstructed = pca.inverse_transform(X_compressed)
   ```

2. **Feature Selection** for high-dimensional data:
   ```python
   # Select top 2 principal components
   pca = PCA(n_components=2)
   X_2d = pca.fit_transform(X_high_dim)  # Visualize in 2D
   explained_variance = pca.explained_variance_ratio_
   ```

#### K-Means Clustering (`Unsupervised machine learning labs/C3_W1_KMeans_Assignment.ipynb`)
**Algorithm**: Partition data into k clusters by minimizing intra-cluster distance

**Business Applications:**
```python
# Customer segmentation
kmeans = KMeans(n_clusters=4, random_state=42)
customer_segments = kmeans.fit_predict(customer_data)

# Analyze segments
for segment in range(4):
    segment_data = customer_data[customer_segments == segment]
    print(f"Segment {segment}: {len(segment_data)} customers")
    print(f"Average spending: ${segment_data['total_spend'].mean():.2f}")
```

#### Anomaly Detection (`Unsupervised machine learning labs/C3_W1_Anomaly_Detection.ipynb`)
**Algorithm**: Statistical model using multivariate Gaussian distribution

**Example Applications:**
```python
# Network security monitoring
def detect_anomalies(X, mu, sigma2, epsilon=1e-6):
    # Compute Gaussian probability density
    p = multivariate_gaussian(X, mu, sigma2)

    # Flag anomalies
    anomalies = X[p < epsilon]
    return anomalies, p

# Server monitoring
server_logs = np.array([[cpu_usage, memory_percent, network_traffic]])
anomalies, probabilities = detect_anomalies(server_logs, mu_train, sigma2_train)
```

#### Recommendation Systems (`Unsupervised machine learning labs/C3_W2_Collaborative_RecSys_Assignment.ipynb`)
**Algorithm**: Matrix factorization for user-item interaction prediction

**Implementation Example:**
```python
# Collaborative filtering
def collaborative_filtering(R, num_features=10, learning_rate=0.1, epochs=100):
    """
    R: Ratings matrix (users x items), 0 where not rated
    Returns: Reconstructed ratings matrix
    """
    U, V = initialize_parameters(R.shape, num_features)

    for epoch in range(epochs):
        # Matrix factorization updates
        U, V = update_parameters(U, V, R, learning_rate)

    predicted_ratings = U @ V.T
    return predicted_ratings

# Movie recommendation
user_movie_ratings = collaborative_filtering(ratings_matrix)
recommended_movies = np.argsort(user_movie_ratings[user_id, :])[::-1][:10]
```

### Natural Language Processing Examples

#### Text Generation (`NLP -N/Generating Text with Neural Networks/`)
**Character-level RNN** (`C3_W4_Lab_3_text_generation.ipynb`)
```python
# Generate Shakespeare-like text
model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(256, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

# Training on Shakespeare corpus
model.fit(X_train, y_train, epochs=50)

# Generate new text
generated_text = generate_text(model, "ROMEO", char2idx, idx2char, length=500)
print(generated_text)
```

#### Next Word Prediction (`NLP -N/perdection the next word/C3W4_Assignment.ipynb`)
**Language Modeling** for autocomplete features
```python
# Simple bigram model
def predict_next_word(current_word, word_counts, bigrams):
    context = word_counts[current_word]
    candidates = [word for word, count in bigrams[(current_word, word)].items()]
    probabilities = np.array([bigrams[(current_word, word)] / context for word in candidates])
    return np.random.choice(candidates, p=probabilities/sum(probabilities))

# Usage
next_word = predict_next_word("machine", word_frequencies, word_pairs)
# Output: probably "learning"
```

### Computer Vision Examples

#### Transfer Learning (`TF/C2_W3_Lab_1_transfer_learning.ipynb`)
**Fine-tuning Pre-trained Models**:
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
base_model.trainable = False

# Add custom classification head
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Train only new layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10)

# Fine-tune base layers
base_model.trainable = True
# Unfreeze last few layers
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Train with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ... )
```

### Hugging Face Transformers Examples

#### Object Detection (`Hugging face -object detection/L8_object_detection.ipynb`)
```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Process image
inputs = processor(images=image, return_tensors="pt")

# Perform detection
with torch.no_grad():
    outputs = model(**inputs)

# Extract results
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

#### Zero-Shot Image Classification (`Hugging face -object detection/L13_Zero_Shot_Image_Classification.ipynb`)
```python
import clip
import torch

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare text and image
classes = ["a cat", "a dog", "a bird", "a car"]
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in classes]).to(device)
image_input = preprocess(image).unsqueeze(0).to(device)

# Encode and compare
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    image_features = model.encode_image(image_input)

# Calculate similarities
text_features /= text_features.norm(dim=-1, keepdim=True)
image_features /= image_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Get prediction
predicted_class = classes[similarity.argmax()]
```

#### Image Captioning (`Hugging face -object detection/L11_image_captioning.ipynb`)
```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load models
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Process image
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate caption
with torch.no_grad():
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4)

# Decode result
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(caption)  # "a cat sitting on a table"
```

## Repository structure (top-level)

- `supervised machine learning labs/` — classic ML labs (cost function, gradient descent, logistic regression, scikit-learn).
- `unsupervised machine learning labs/` — PCA, KMeans, anomaly detection, recommender systems.
- `NLP -N/` — NLP assignments and IMDB notebook examples; includes `unittests.py` used for automatic grading.
- `Image_data_preprocessing/` and `week 4.lab3 image_data_preprocessing/` — image preprocessing labs and horse-or-human examples.
- `Week 4 Multi-class _Classification/` — examples and assignments for multi-class classification (TensorFlow).
- Various standalone notebooks at the repository root (e.g., `C1_W2_Lab_1_beyond_hello_world.ipynb`, `C2_W3_Lab_1_transfer_learning.ipynb`, `C2W2_cat&dog.ipynb`).


## 🛠️ Technical Requirements and Setup

### Dependencies
Core libraries used across notebooks:

```
Python 3.8+
jupyter/notebook     # Interactive development
numpy               # Numerical computations
pandas              # Data manipulation
matplotlib/seaborn  # Visualization
scikit-learn        # Classical ML algorithms
tensorflow/keras    # Deep learning
pillow (PIL)       # Image processing
```

### Quick Start Guide

1. **Environment Setup**
   ```powershell
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate

   # Install dependencies
   python -m pip install -r requirements.txt
   ```

2. **Launch Jupyter**
   ```powershell
   jupyter lab  # For JupyterLab interface
   # or
   jupyter notebook  # For classic Notebook interface
   ```

3. **Dataset Preparation**
   - Download required datasets (see individual notebook instructions)
   - Image datasets are in respective `/data/` folders
   - Ensure sufficient disk space for image processing tasks

4. **Notebook Selection**
   - Choose notebooks based on your learning goals
   - Follow the recommended sequence within each section
   - Execute cells in order for proper initialization

## 🎓 Learning Outcomes

After completing this repository's notebooks, you will be able to:

1. **Machine Learning Fundamentals**
   - Implement gradient descent and optimization algorithms from scratch
   - Understand and apply regularization techniques
   - Master feature scaling and preprocessing

2. **Deep Learning & Neural Networks**
   - Build and train neural networks using TensorFlow/Keras
   - Implement transfer learning for efficient model development
   - Design custom architectures for specific problems

3. **Natural Language Processing**
   - Process and analyze text data
   - Build sentiment analysis models
   - Implement text classification systems

4. **Computer Vision**
   - Process and augment image data
   - Build image classification models
   - Apply transfer learning to vision tasks

5. **Practical Skills**
   - Write clean, documented code
   - Implement unit tests for ML systems
   - Build end-to-end ML pipelines

## 🏆 Project Highlights

### Advanced Implementations
- Custom optimization algorithms
- Neural network architectures
- Recommendation systems
- Anomaly detection systems

### Industry-Ready Skills
- Data preprocessing pipelines
- Model evaluation techniques
- Production-ready code practices
- Performance optimization

### Real-World Applications
- Medical diagnosis systems
- Customer behavior prediction
- Content recommendation
- Image recognition systems


## 📊 Datasets

This repository uses various datasets for different learning objectives:

- **Binary Classification**: Medical diagnosis dataset
- **Image Classification**: Horse-or-human dataset
- **Text Classification**: IMDB reviews dataset
- **Recommendation**: Movie ratings dataset

Datasets are either included in the repository or can be downloaded using provided scripts.

## 📝 Note on Large Files

Some notebooks work with large datasets stored in `/data/` folders. Consider:
- Using Git LFS for version control
- Downloading datasets on demand
- Following storage requirements in notebook headers

## 📜 License

This project is provided under the MIT License — see `LICENSE`.

## 🤝 Contributing Guidelines

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📫 Contact

For questions and feedback:
- Create an issue in the repository
- Start a discussion
- Contribute improvements

## 📚 Usage Examples Summary

Here's a quick reference for the most commonly used algorithms and their practical implementations:

### Classification
- **Logistic Regression**: Binary classification (`Supervised machine learning labs/C1_W3_Logistic_Regression.ipynb`)
- **Neural Networks**: Multi-class classification (`Week 4 Multi-class _Classification/`, `TF/`)

### Regression
- **Linear Regression**: Gradient descent optimization (`Supervised machine learning labs/C1_W3_Lab06_Gradient_Descent_Soln.ipynb`)
- **Polynomial Regression**: Feature engineering (`_FeatEng_PolyReg_Soln.ipynb`)

### Clustering & Dimensionality Reduction
- **K-Means**: Customer segmentation (`Unsupervised machine learning labs/C3_W1_KMeans_Assignment.ipynb`)
- **PCA**: Data compression and visualization (`Unsupervised machine learning labs/C3_W2_Lab01_PCA_Visualization_Examples.ipynb`)

### Computer Vision
- **Transfer Learning**: MobileNetV2 fine-tuning (`TF/C2_W3_Lab_1_transfer_learning.ipynb`)
- **Object Detection**: DETR model (`Hugging face -object detection/L8_object_detection.ipynb`)
- **Image Classification**: Zero-shot with CLIP (`Hugging face -object detection/L13_Zero_Shot_Image_Classification.ipynb`)

### Natural Language Processing
- **Text Generation**: Character-level RNN (`NLP -N/Generating Text with Neural Networks/C3_W4_Lab_3_text_generation.ipynb`)
- **Next Word Prediction**: Language modeling (`NLP -N/perdection the next word/C3W4_Assignment.ipynb`)

### Anomaly Detection & Recommendations
- **Statistical Anomalies**: Gaussian models (`Unsupervised machine learning labs/C3_W1_Anomaly_Detection.ipynb`)
- **Collaborative Filtering**: Matrix factorization (`Unsupervised machine learning labs/C3_W2_Collaborative_RecSys_Assignment.ipynb`)

---
_Last updated: October 5, 2025_ - Last updated content: October 14, 2025_


