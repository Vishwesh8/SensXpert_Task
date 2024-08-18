from utilities import *

# Get current directory
cwd = os.getcwd()
# Get parent directory
task_dir = os.path.abspath(os.path.join(cwd, os.pardir))
# Get directory of generated data
generated_data_dir = os.path.join(task_dir, "generated_data")

with open(os.path.join(generated_data_dir, "x_train"), 'rb') as f:
    X_train = pickle.load(f)

with open(os.path.join(generated_data_dir, "x_val"), 'rb') as f:
    X_val = pickle.load(f)

with open(os.path.join(generated_data_dir, "y_train"), 'rb') as f:
    y_train = pickle.load(f)

with open(os.path.join(generated_data_dir, "y_val"), 'rb') as f:
    y_val = pickle.load(f)

# Finding the dimensions of input data
sequence_length = X_train.shape[1]
feature_dim = X_train.shape[2]

input_shape = (sequence_length, feature_dim)

lstm_model = create_lstm_model(input_shape)
gru_model = create_gru_model(input_shape)
cnn_model = create_cnn_model(input_shape)
tcn_model = create_tcn_model(input_shape)

# Select required model
model = lstm_model
# model = gru_model
# model = cnn_model
# model = tcn_model

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

start_time = time.time()
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2000,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)
end_time = time.time()
print(f"\nTotal training time for LSTM model = {end_time-start_time} seconds")

# Plotting training and validation error during training
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plotting training and validation MAE during training
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Save model
models_dir = os.path.join(task_dir, "models")

model_dir = os.path.join(task_dir, "LSTM")
# model_dir = os.path.join(task_dir, "GRU")
# model_dir = os.path.join(task_dir, "CNN")
# model_dir = os.path.join(task_dir, "TCN")

# model.save(generated_data_dir=os.path.join(model_dir, "lstm_model.h5"))
# model.save(generated_data_dir=os.path.join(model_dir, "gru_model.h5"))
# model.save(generated_data_dir=os.path.join(model_dir, "cnn_model.h5"))
# model.save(generated_data_dir=os.path.join(model_dir, "tcn_model.h5"))

print('Model Saved!')
