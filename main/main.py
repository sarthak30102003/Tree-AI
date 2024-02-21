import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 256, 256
batch_size = 32
epochs = 100
train_data_dir = "O:\RADON\Train"
validation_data_dir = "O:\RADON\Validation"
test_data_dir ="O:\RADON\Test

class_mapping = {
    0: ('Generic', 'Healthy'),
    1: ('Generic', 'Powdery'),
    2: ('Generic', 'Rusty'),
    3: ('CORN', 'Rust'),
    4: ('CORN', 'Gray SPot'),
    5: ('CORN', 'Healthy'),
    6: ('CORN', 'Leaf Blight'),
    7: ('Potato', 'Early Blight'),
    8: ('Potato', 'Healthy'),
    9: ('Potato', 'Late Blight'),
    10: ('SugarsCane', 'Bacteria Blight'),
    11: ('SugarCane', 'Healthy'),
    12: ('SugarCane', 'RedRot'),
    13: ('Wheat', 'Brown Rust'),
    14: ('Wheat', 'Healthy'),
    15: ('Wheat', 'Yellow Rust'),
}

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(class_mapping)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc}')

model.save('plant_disease_model.h5')

predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)
for i, prediction in enumerate(predictions):
    predicted_class = tf.argmax(prediction).numpy()
    species, disease = class_mapping[predicted_class]
    print(f"Sample {i+1}: Predicted Class - {predicted_class}, Species - {species}, Disease - {disease}")
