from keras.models import Sequential
from keras.layers import Dense, Activation
import csv
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def ReadCsvFile(path):
    # Reads a csv file and returns 1d numpy array of labels, 2d numpy array of data

    labels = []
    data = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        line_count = 0
        for row in reader:
            if line_count == 0:
                # Column names
                labels = row
            else:
                # Data
                data.append(row)
            line_count += 1
    
    labels = np.array(labels)
    data = np.array(data)
    
    return labels, data

    

# Load data
labels, train_x = ReadCsvFile("train.csv")
train_y = train_x[:, 1]

_, test_x = ReadCsvFile("test.csv")

# Convert male/female to numbers
train_x[:, 4] = np.where(train_x[:, 4] == "male", 1, 0)
test_x[:, 3] = np.where(test_x[:, 3] == "male", 1, 0)

# Set a random age where none exist
train_x[:, 5] = np.where(train_x[:, 5] == '', random.random() * 60, train_x[:, 5])
test_x[:, 4] = np.where(test_x[:, 4] == '', random.random() * 60, test_x[:, 4])

# Save test passenger IDs for submission later
test_passenger_ids = np.reshape(test_x[:, 0], (test_x.shape[0], 1))

# Filter any uninteresting data
train_x = train_x[:, [2, 4, 5, 6, 7]]
test_x = test_x[:, [1, 3, 4, 5, 6]]
labels = labels[[2, 4, 5, 6, 7]]

# Convert to float
train_x = train_x.astype(np.float)
train_y = train_y.astype(np.float)
test_x = test_x.astype(np.float)

# Standardize the value ranges
scaler = StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

# Define binary classification model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=train_x.shape[1]))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_x, train_y, epochs=20, batch_size=32)

# Run the model against the test data
predict_y = model.predict(test_x)
predict_y = np.around(predict_y)
predict_y = predict_y.astype(np.integer)

# Write our predictions to a csv file
csv_predict = np.concatenate((test_passenger_ids, predict_y), axis=1)
csv_predict = np.concatenate((np.reshape(["PassengerId", "Survived"], (1, 2)), csv_predict))
with open('prediction.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csv_predict)
csvFile.close()