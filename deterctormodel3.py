# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.mixed_precision import set_global_policy

# # Enable mixed precision for faster training
# set_global_policy('mixed_float16')

# # Constants
# IMG_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS_TOP = 5       # Train top layers first
# EPOCHS_FINE = 10     # Fine-tune entire model
# DATASET_PATH = '/kaggle/input/brain-tumor-mri-dataset'

# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# train_generator = train_datagen.flow_from_directory(
#     DATASET_PATH + '/Training',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     DATASET_PATH + '/Training',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # Load VGG16 base
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base_model.trainable = False  # Freeze base initially

# # Add top layers
# x = Flatten()(base_model.output)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# output = Dense(4, activation='softmax', dtype='float32')(x)  # Force output to float32 for mixed precision

# model = Model(inputs=base_model.input, outputs=output)

# # Compile
# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Step 1: Train top layers first
# history_top = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS_TOP,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_steps=val_generator.samples // BATCH_SIZE,
#     verbose=1
# )

# # Step 2: Fine-tune some base layers
# for layer in base_model.layers[-4:]:  # Unfreeze last 4 conv layers
#     layer.trainable = True

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history_fine = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS_FINE,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_steps=val_generator.samples // BATCH_SIZE,
#     verbose=1
# )
