# # Step 1: Import Libraries
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# # Step 2: Set dataset path
# base_dir = '/kaggle/input/brain-tumor-mri-dataset/'

# # Step 3: Count total MRI scans in each folder
# classes = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
# total_images = 0

# for folder in classes:
#     folder_path = os.path.join(base_dir, folder)
#     # Count only image files (skip hidden files)
#     count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#     total_images += count
#     print(f"Folder '{folder}': {count} images")

# print(f"\nTotal MRI scans: {total_images}")

# # Step 4: Safely visualize one image from each class
# plt.figure(figsize=(8,4))
# for i, folder in enumerate(classes):
#     folder_path = os.path.join(base_dir, folder)
#     for file in os.listdir(folder_path):
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(folder_path, file)
#             img = cv2.imread(img_path)
#             if img is not None:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 plt.subplot(1,2,i+1)
#                 plt.imshow(img)
#                 plt.title(folder)
#                 plt.axis('off')
#                 break  # stop after first valid image
# plt.show()

# # Step 5: Image preprocessing
# IMG_SIZE = 150
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     horizontal_flip=True,
#     zoom_range=0.2
# )

# train_generator = datagen.flow_from_directory(
#     base_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )

# val_generator = datagen.flow_from_directory(
#     base_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )

# # Step 6: Build CNN model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# # Step 7: Train the model
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator
# )

# # Step 8: Plot training history
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title('Accuracy')

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title('Loss')

# plt.show()

# # Step 9: Save the model
# model.save('brain_tumor_model.h5')
