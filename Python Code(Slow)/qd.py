import pandas as pd
import numpy as np
import pyximport;
from sklearn.model_selection import train_test_split

pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import TsetlinMachine

# Parameters for the Tsetlin Machine
T = 40
s = 27
number_of_clauses = 12000
states = 100

# Parameters of the pattern recognition problem
number_of_features = 706
number_of_classes = 6

# Training configuration
epochs = 200

df1 = pd.read_csv('QD_BIN_FINAL.csv', header = None)
df2 = pd.read_csv('label.csv', header = None)
x = df1.iloc[:,:].values
y = df2.iloc[:,:].values
y = np.reshape(y, len(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.084, shuffle=False)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.astype(dtype=np.int32)
y_train = y_train.astype(dtype=np.int32)
x_test = x_test.astype(dtype=np.int32)
y_test = y_test.astype(dtype=np.int32)

print('--------------Training Started--------------------------')
tsetlin_machine = TsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)

tsetlin_machine.fit(x_train, y_train, y_train.shape[0], epochs=epochs)

print("Accuracy on test data (no noise):", tsetlin_machine.evaluate(x_test, y_test, y_test.shape[0]))
print("Accuracy on training data (40% noise):", tsetlin_machine.evaluate(x_train, y_train, y_train.shape[0]))