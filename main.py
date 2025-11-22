from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing: rescale pixel values
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images
train_data = train_datagen.flow_from_directory(
    'archive/Training',           # path to your training folder
    target_size=(64, 64),         # resize images
    batch_size=32,                # how many images to process at once
    class_mode='categorical'      # multi-class classification
)

# Load testing images
test_data = test_datagen.flow_from_directory(
    'archive/Testing',            # path to your testing folder
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Get class names in the correct order (alphabetical)
class_names = sorted(train_data.class_indices.keys())
print(f"Classes found: {class_names}")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, pituitary, notumor
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=10, validation_data=test_data)
# Save the trained model
model.save('tumor_model.h5')
print("Model saved successfully.")
import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess the image
img = image.load_img('archive/Testing/meningioma/Te-me_0296.jpg', target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Map index to class name (class_names already defined above from train_data)
print("Predicted Tumor Type:", class_names[predicted_class])
print("Confidence Scores:", prediction[0])
for i, (name, score) in enumerate(zip(class_names, prediction[0])):
    print(f"  {name}: {score:.4f}")

# Visualize predictions
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

plt.bar(class_names, prediction[0])
plt.title("Prediction Confidence")
plt.xlabel("Tumor Type")
plt.ylabel("Confidence")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('prediction_results.png')
print("\nPrediction visualization saved as 'prediction_results.png'")
plt.close()  # Close the figure to free memory