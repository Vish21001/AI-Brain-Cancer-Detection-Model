# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.optimizers import Adam

# # Force GPU (optional)
# # Comment this out if you want GPU usage
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment to force CPU

# # Paths and parameters
# DATASET_PATH = "/kaggle/input/brain-tumor-mri-dataset"
# TRAIN_DIR = os.path.join(DATASET_PATH, "Training")
# VAL_DIR = os.path.join(DATASET_PATH, "Testing")

# IMG_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS = 15
# LEARNING_RATE = 1e-4
# NUM_CLASSES = 4

# # Data Generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=True
# )

# val_generator = val_datagen.flow_from_directory(
#     VAL_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False
# )

# # Model Setup
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base_model.trainable = False  # Freeze base layers initially

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

# model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# # Train the model
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS,
#     verbose=1
# )

# # Fine-tune (optional)
# # Unfreeze last layers for fine-tuning
# base_model.trainable = True
# for layer in base_model.layers[:-30]:  # freeze most layers, fine-tune last 30
#     layer.trainable = False

# model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# history_fine = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10,
#     verbose=1
# )
