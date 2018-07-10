from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.misc import imshow
from scipy.misc import imresize
from scipy.fftpack import fft

from scipy import signal
import scipy.cluster.hierarchy as hcluster


files = Path('../Raw').glob('*.tif')


#Put in a 1D array along with cutoff freq, order return the filtered array of same shape
def filter_data(in_data, fc, order, type):
  #Generate a "time" variable
  t = range(len(in_data))
  
  #highpass butterworth
  b,a = signal.butter(order, fc, type)
  
  #Apply filter to data
  # zi = signal.lfilter_zi(b,a)
  # z,_ = signal.lfilter(b,a,in_data,zi=zi*in_data[0])
  
  #Applying filtfilt, forwards and backwards for zero phase
  y = signal.filtfilt(b,a,in_data)
  # plt.figure
  # plt.plot(t,in_data,'b')
  # plt.plot(t,y,'r--')
  # plt.grid(True)
  return y

  
#~~~~~~~~~~~~~Parameters~~~~~~~~~~~~~~~~~~~~~
#1048 * 1328
threshold = 130
scale = 4
max_steps = 1500


counts = []
pixelGraph = []

h = int(1048/scale)
w = int(1328/scale)
shape = (h,w)
# shape3 = (1048,1328,3072)
shape3 = (h,w,max_steps)
last = np.zeros(shape)
findChange = np.zeros(shape)
time_data = np.zeros(shape3)


pix_time = [[[]]]


#FFT setup
N = max_steps
T = 1.0/(120) #Not sure what the period should be...





#im = Image.fromarray(findChange)
#im.show()

# ~~~~~~~~~~~Importing Data~~~~~~~~~~~~~~~~~
#Bring Video data to 3D array
for num,f in enumerate(files):

  if num >= max_steps:
    break
    
  #Open image
  im = Image.open(f)#,dtype='int64')  
  
  # im.show()
  
  
  #Create np array
  cur_temp = np.array(im)
  
  #Resizing
  cur = imresize(cur_temp,1/scale)
  
  
  
  time_data[:,:,num] = cur
  if num % 50 == 0:
    print('Step:', num)
  
  if num % 200 == 0:
    im = Image.fromarray(cur)
    im = im.convert('RGB')
    im.save('Video/' + str(num) + 'Pic.png',"PNG")
    
#Get total timesteps
steps = range(len(time_data[0,0]))
    
print('Finished importing data!')

#FFT Data
fft_data = np.zeros(shape3, dtype = complex)

print('FFT Time...')


 

# ~~~~~~~~~~~~~~FFT~~~~~~~~~~~~~~~~~~~~~
for i in range(shape[0]):
  for j in range(shape[1]):
    # print(time_data[i,j])
    # fft_data[i,j] = fft(time_data[i,j]/255)
    fft_data[i,j] = np.fft.fft(time_data[i,j]/255)
    
    
    # if j%30 == 0 and i > 100:
      # n = fft_data[i,j].size
      # delta_t = 1/120
      # freqs = np.fft.fftfreq(n,delta_t)
      # plt.plot(freqs[:n//2],fft_data[i,j][:n//2],'r')
      # plt.axis([0,60,0,.5])
      # plt.show()
      # 
      
    
  if i % 20 == 0:
    print('FFT step',i,'of',shape[0])
    
    
print('FFT Complete!')
    
    
    
  
    
    
time = range(len(counts))






#~~~~~~~~~~~~~~~~~~~~FFT Display~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
key_in = 136

im = Image.fromarray(time_data[:,:,0])
# for _ in range(max_steps):
while(1):
  key_in = int(input('\nPick a FFT Value(1 to '+ str(len(fft_data[0][0])) + ' or 0 to quit, -1 to save): '))
  
  if key_in == 0:
    break
  if key_in == -1:
    str_in = input('\nEnter filename(no filetype): ')
    # im = Image.fromarray(neural_map)
    str_in += '.png'
    # im = im.convert('RGB')
    im.save(str_in ,"PNG")
 
  
      
        
  im = Image.fromarray(time_data[:,:,0])
  im = im.convert('RGB')
  pix = im.load()

  for i in range(shape[0]):
    for j in range(shape[1]):
      # print('i',i,'j',j)
      # print(fft_data[i,j, key_in])
      
      pix[j,i] = (fft_data[i,j,key_in]*pix[j,i][0],pix[j,i][1],pix[j,i][2])
      
    # cur_name = 'FFT_Sweep/FFT_' + str(key_in) + '.png'
    # im.save(cur_name, "PNG")
  im.show()
    
  # break
  
 
#~~~~~~~~~~~~~~~~~~Neural Mapping Threshold~~~~~~~~~~~~~~~~~~~

key_in = 1
neural_map = np.zeros(shape)

#Load RGB pixel values
pix = im.load()
print('pix:',pix[j,i][0])
while(1):
  light_thresh = int(input('\nPick a light threshold(1-255, 0 to quit, -1 to save): '))
   
  if light_thresh == 0:
    break
  
  
  #Check if red channel is above thresh
  for i in range(shape[0]):
    for j in range(shape[1]):
      if pix[j,i][0] > light_thresh:
        neural_map[i][j] = 255
      
        
  map = Image.fromarray(neural_map)
  map.show()
  
 
 

#~~~~~~~~~~~~~~~~~~Clustering~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Holding data indices -- Each row holds [row, column]
cluster_data = np.zeros((1,2),dtype='int')
centroid_dict = dict()

key_in_clust = 10
while(1):
  key_in_clust = float(input('\nPick a clustering threshold(0 to quit): '))
  
  if key_in_clust == 0:
    break
  
  c_threshold = key_in_clust
  
  #Reset cluster data to zero
  cluster_data = np.zeros((1,2),dtype='int')
  
  #Loop through pixels, append indices above threshold
  for i in range(shape[0]):
      for j in range(shape[1]):
        if pix[j,i][0] >100:
          cluster_data = np.vstack([cluster_data,[i,j]])
          
         
  
  print('Total Above Threshold =',len(cluster_data))
  input('Cluster? Press enter:')
  print('Clustering...')
  clusters = hcluster.fclusterdata(cluster_data, c_threshold, criterion='distance')
  
  
  #Plot data
  plt.scatter(*np.transpose(cluster_data), c=clusters)
  diff_clusters = set(clusters)
  centroid_dict = dict.fromkeys(diff_clusters,(0,0))
  
  #Iterate through groups, average indices
  for group in diff_clusters:
    count = 0
    
    for num,(i,j) in enumerate(cluster_data):
      if clusters[num] == group:
        centroid_dict[group] = (centroid_dict[group][0] + i, centroid_dict[group][1] + j)
        count += 1
    
    #Average and annotate plot
    centroid_dict[group] = tuple(cur/count for cur in centroid_dict[group])
    plt.annotate(str(group),xy=centroid_dict[group])
  
  
  
  
  
  #plt.annotate(clusters,
  plt.axis("equal")
  title = "threshold: %f, number of clusters: %d" % (c_threshold, len(set(clusters)))
  plt.title(title)
  plt.show()
  # break
  

#~~~~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spike_threshold = 15

spike_list = [[]]
used = []


key_in = 1

while(1)
  if key_in == 0:
    break
  
  #Take input, remove whitespace
  key_in = int(input('\nPick clusters to plot(Comma separated): '))
  key_in = key_in.replace(' ','')
  
  
# while key_in != 0:
  # key_in = int(input('\nPick Raster Plot Threshold: '))
  # spike_threshold = key_in
  # spike_list = [[]]
  
  change to cluster_data
  # for num,(i,j) in enumerate(hard_list):
    # temp = []
    
    
    # if True:
    if clusters[num] not in used:
      Loop through and append all spikes
      # for s in range(max_steps):
        print(time_data[i,j,s])
        print('num',clusters[num])
        print('used',used)
        # if time_data[i,j,s] > spike_threshold:
          # temp.append(s)
          
          print('Did it')
        
      # if len(temp) >= 1:
        # spike_list.append(temp)
        # used.append(clusters[num])
        
    
    
# print(spike_list[:20])

  print('Length of List:', len(spike_list))

  pad = len(max(spike_list, key=len))
  spikes = np.array([i + [0]*(pad-len(i)) for i in spike_list])
  # np.random.shuffle(spikes)

  plt.eventplot(spikes, linelengths = .6)
  plt.title('Calcium Imaging Activation')
  plt.xlabel('Steps')
  plt.ylabel('Pixels above threshold')
  plt.show()
  


fig, ax = plt.subplots()
# ax.plot(time,counts)
np.random.shuffle(cluster_data)
# for num,(j,i) in enumerate(cluster_data):
  # ax.plot(steps,time_data[i,j])
  # if num > 8:
    # break
    
xx = (time_data[148,146] + time_data[148,145] + time_data[147,146] + time_data[147,145])/4

xx = (xx/np.amax(xx))



ax.plot(steps,xx)
# ax.plot(steps,time_data[148,145])
# ax.plot(steps,time_data[147,145])
# ax.plot(steps,time_data[147,146])



# ax.plot(steps,time_data[100,240])


ax.set(xlabel='Steps',ylabel = 'Magnitude',title = 'Activation of pixels vs. Time')
ax.grid()
#fig.savefig('ActivationFull.png')
plt.show()
