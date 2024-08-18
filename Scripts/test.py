from utilities import *

# Get current directory
cwd = os.getcwd()
# Get parent directory
task_dir = os.path.abspath(os.path.join(cwd, os.pardir))
# Get directory of generated data
generated_data_dir = os.path.join(task_dir, "generated_data")

# Loading test data
with open(os.path.join(generated_data_dir, "x_test"), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(generated_data_dir, "y_test"), 'rb') as f:
    y_test = pickle.load(f)

# Loading models

models_dir = os.path.join(task_dir, "models")

# model_dir = os.path.join(models_dir, "LSTM")
# model_dir = os.path.join(models_dir, "GRU")
# model_dir = os.path.join(models_dir, "CNN")
model_dir = os.path.join(models_dir, "TCN")

# model = load_model(os.path.join(model_dir, "lstm_model.h5"))
# model = load_model(os.path.join(model_dir, "gru_model.h5"))
# model = load_model(os.path.join(model_dir, "cnn_model.h5"))
model = load_model(os.path.join(model_dir, "tcn_model.h5"), custom_objects={'TCN': TCN})

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print("Model Evaluation - ")
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

predictions = model.predict(X_test)

plt.scatter(range(25), predictions[:25], label='Prediction')
plt.scatter(range(25), y_test[:25], label='Actual')
plt.title('TCN Model - Predictions Vs. Actual')
plt.ylabel('Delta-T (minutes)')
plt.xlabel('Test Cycles')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
