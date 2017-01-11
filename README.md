# ball-mill-LSTM
The program is in Tensorflow and python, based in DNN-LSTM
is a regression for a ball mill machine 
This is a regression for the sensor that measures the level 
of material inside the ball mill, fill level, we used two types of sensors, 
sound and accelerometer.
For the most interested there is a paper with all the information about this project and code

The first step is the creation of the database with the program, 'data_generator.py', then with the programs
'MPL_encoding_vibration.py' and ' 	MPL_encodig_vibration.py' the dimensional reduction is performed and they obtain the main
characteristics of the Data base, and finally with the program 'LSTM_regression.py' the regression is performed.

Everything is done on an HP computer, with 
Processor: Intel® Core ™ i7-6700K CPU @ 4.00GHz × 8
Menory: 15.6 GiB
Graphics: GeForce GTX 980 Ti / PCIe / SSE2 - 6GiB

The whole process takes about 4 hours
