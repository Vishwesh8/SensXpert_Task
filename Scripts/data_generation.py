from utilities import *

# Get current directory
cwd = os.getcwd()
# Get parent directory
task_dir = os.path.abspath(os.path.join(cwd, os.pardir))
# Get data directory
data_dir = os.path.join(task_dir, "data")

files = glob.glob(data_dir + "/*.csv")
dataset = np.empty(2, dtype='object')

start_time = time.time()
# loop over the list of csv files
for f in files:
    # read the csv file
    df = pd.read_csv(f, encoding='unicode_escape', sep="\s|:", engine='python')

    # Keeping only required columns and converting data to required format
    df = df.iloc[:, :4]
    df = df.applymap(lambda x: float(x.replace(',', '.').replace('E', 'e')))

    # Finding critical points and creating dataset
    cp2, cp3, cp4 = critical_points(df)
    xcp23 = find_train_data(df, cp2, cp3)
    # Removing temperature values as its distribution between CP2 and CP3 should not affect the values of delta_t
    # Because, it is almost constant for all the measurements
    xcp23 = xcp23.iloc[:, 1:4]
    delta_t = find_delta_t(df, cp2, cp4)

    mean_time_cp2_cp3 = df['time_/min'].iloc[cp2:cp3 + 1].mean()
    xcp23 = data_transformation(xcp23, mean_time_cp2_cp3)

    # Converting xcp23 into numpy array as it will easier for training later
    xcp23 = np.array(xcp23)
    temp_data = [xcp23, delta_t]
    temp_data = np.array(temp_data, dtype='object')

    # Stacking data for this cycle in the dataset
    dataset = np.vstack((dataset, temp_data))

end_time = time.time()
print(f"\nTotal time taken for data preparation = {end_time - start_time} seconds")
dataset = dataset[1:]
print(f"Dimensions of the available dataset are = {dataset.shape}")

# Finding max and min length of xcp23 for all the measurements
max_length = 0
min_length = 1000000
for x in dataset[:, 0]:
    if x.shape[0] > max_length:
        max_length = x.shape[0]
    if x.shape[0] < min_length:
        min_length = x.shape[0]
print(f"Longest xcp23 = {max_length}, Shortest xcp23 = {min_length}")

# As the input data (xcp23) has variable length for different measurements, padding the data to bring uniformity
# Randomly shuffling dataset to remove any sequential biases
np.random.shuffle(dataset)

# Separating input and target values
X = dataset[:, 0]
y = dataset[:, 1]

# Pad sequences to the same length as the inputs are of variable lengths
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', dtype='float32')
print(f"Dataset dimensions - {X_padded.shape}")

# Converting targets to numpy array
y = np.asarray(y).astype(np.float32)
print(f"Target dimensions - {y.shape}")

# Finding the dimensions of input data
num_samples = X_padded.shape[0]
sequence_length = X_padded.shape[1]
feature_dim = X_padded.shape[2]

# Splitting into training and testing sets
train_test_split_index = int(0.8 * num_samples)
X_train, X_test = X_padded[:train_test_split_index], X_padded[train_test_split_index:]
y_train, y_test = y[:train_test_split_index], y[train_test_split_index:]
print(f"Test set dimensions - {X_test.shape}, {y_test.shape}")

# Splitting into training and validation sets
train_val_split_index = int(0.8 * X_train.shape[0])
X_train, X_val = X_train[:train_val_split_index], X_train[train_val_split_index:]
y_train, y_val = y_train[:train_val_split_index], y_train[train_val_split_index:]
print(f"Train set dimensions - {X_train.shape}, {y_train.shape}")
print(f"Validation set dimensions - {X_val.shape}, {y_val.shape}")

# Saving training, testing and validation data
generated_data_dir = os.path.join(task_dir, "generated_data")
with open(os.path.join(generated_data_dir, "x_train"), 'wb') as f:
    pickle.dump(X_train, f)

with open(os.path.join(generated_data_dir, "x_test"), 'wb') as f:
    pickle.dump(X_test, f)

with open(os.path.join(generated_data_dir, "x_val"), 'wb') as f:
    pickle.dump(X_val, f)

with open(os.path.join(generated_data_dir, "y_train"), 'wb') as f:
    pickle.dump(y_train, f)

with open(os.path.join(generated_data_dir, "y_test"), 'wb') as f:
    pickle.dump(y_test, f)

with open(os.path.join(generated_data_dir, "y_val"), 'wb') as f:
    pickle.dump(y_val, f)



