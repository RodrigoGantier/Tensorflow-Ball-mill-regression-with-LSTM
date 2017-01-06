#coding:utf8
'''
Created on Jan 6, 2017

@author: lab
'''

import numpy as np
from scipy.signal import welch
import scipy.signal as signal
import scipy.io as sio


###############################################


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

if __name__ == "__main__":
    
    
    fs = 51200
    frame_len = int(fs/2)
    sectio_size = 131072
    contador=0
    
    file_name = list('/media/lab/软件/采集的数据/湿式球磨机数据采集/第一次实验/balltest/1/1.mat')
    audio_file = list('data1')
    
    data_x = np.zeros([140, 8, 22 ,550])
    data_y = np.zeros([140, 8, 22])
    for i in range(1,140):
        file_name[-7]= str(i)
        if i==20:
            print 'alto'
        for ii in range(1,9):
            file_name[-5]= str(ii)
            file_data = sio.loadmat("".join(file_name))
            train_x = file_data["data_path"]
   
            audio = train_x.flatten()-train_x.flatten().mean()
            audio = np.split(audio, range(sectio_size, audio.shape[0],sectio_size))[:22]
            audio = signal.decimate(audio, 2)
            data = welch(x = audio, fs= fs/2, window=np.hamming(frame_len), nperseg = frame_len, nfft=frame_len)
            #plt.imshow(-np.maximum(-0.05*data[1].max(), -1*data[1]), aspect='auto')
            #media = data[1].mean(0)
            #aa = np.maximum(0, media-media.max()*0.05)
            #plt.plot(np.minimum(0.0000000000002, aa)*50000000000000)
            data = pooling_mean(data[1][:,:11000], ventana=20)
            data_x[i, ii-1, :, :] = data
            data_y[i, ii-1, :22]  = i
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
        
