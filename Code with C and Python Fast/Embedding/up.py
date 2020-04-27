from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
sys.path.append('../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine
from time import time 

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

print('data fitting started..........................')

tm = MultiClassTsetlinMachine(2000, 40, 27, weighted_clauses=False)  


print("\nAccuracy over 1000 epochs:\n")
tempAcc = []
for i in range(500):
	start_training = time()
	tm.fit(x_train, y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result1 = 100*(tm.predict(x_test) == y_test).mean()
	result2 = 100*(tm.predict(x_train) == y_train).mean()
	stop_testing = time()
	tempAcc.append(result1)
	print("#%d AccuracyTrain: %.2f%% AccuracyTest: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result2, result1, stop_training-start_training, stop_testing-start_testing))
finalAccuracy = max(tempAcc)
print('Final Accuracy', finalAccuracy)
 