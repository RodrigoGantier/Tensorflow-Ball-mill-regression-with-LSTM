#coding:utf8
'''
Created on Jan 9, 2017

@author: lab
'''
import numpy as np
from scipy.signal import welch
import scipy.signal as signal
import scipy.io as sio

def pooling_mean(data_x, ventana):
    contador = 0
    limite = data_x.shape[1]%ventana
    data_final_x = np.zeros([data_x.shape[0], int(data_x.shape[1]/ventana)])
    if limite == 0:
        data_xx = np.split(data_x, data_x.shape[1]/ventana, axis=1)
    else:
        data_xx = np.split(data_x[:,:-limite], data_x.shape[1]/ventana, axis=1)
    for i in data_xx:
        data_final_x[:,contador] = i.mean(1)
        contador+=1
    return data_final_x

def main():

    fs = 51200 #sample frequency
    frame_len = int(fs/2) #hamming wondow freme lenght
    sectio_size = 131072  #sample division
    contador=0
    
    file_name = list('/media/lab/软件/湿式球磨机数据/第1次实验/balltest/1/1.mat')
    
    data_x = np.zeros([140, 8, 22 ,550])  #data storage matrix
    data_y = np.zeros([140, 8, 22])       #label storage matrix
    for i in range(1,140):                #choose folder
        file_name[-7]= str(i)
        for ii in range(1,9):             #choose archive
            file_name[-5]= str(ii)
            file_data = sio.loadmat("".join(file_name)) #extract data
            train_x = file_data["data_path"]            
   
            audio = train_x.flatten()-train_x.flatten().mean() #rest the mean
            audio = np.split(audio, range(sectio_size, audio.shape[0],sectio_size))[:22] #dived the data in 22 parts
            data = welch(x = audio, fs= fs/2, window=np.hamming(frame_len), nperseg = frame_len, nfft=frame_len) #extrat the psv
            data = pooling_mean(data[1][:,:11000], ventana=20) #cutoff the freq beetween 11000 and 12800 and mean pooling 
            data_x[i, ii-1, :, :] = data   #storage the data matrix [:, 5, :, :] and [:, 7, :, :] are vibration and audio signals
            data_y[i, ii-1, :22]  = i      #create the labels
            contador+=1
            print "Iteracion numero: "+repr(contador)
            
    np.save("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_data_x.npy", data_x) 
    np.save("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_data_y.npy", data_y)
    
    #train test set divition
    train_x = data_x[:, :, range(0, 22, 2), :]
    train_y = data_y[:, :, range(0, 22, 2)]
        
    test_x = data_x[:, :, range(1, 22, 2),:]
    test_y= data_y[:, :, range(1, 22, 2)]
        
        
    np.save("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_train_x.npy", train_x) #guardar la matriz de datos finales
    np.save("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_train_y.npy", train_y) #guardar la matriz de datos finales
    np.save("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_test_x.npy", test_x) #guardar la matriz de datos finales
    np.save("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_test_y.npy", test_y) #guardar la matriz de datos finales
        

if __name__ == "__main__":
    main()
    
    