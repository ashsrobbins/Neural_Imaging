from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.misc import imshow
from scipy.misc import imresize

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
scale = 2
max_steps = 600


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

#Filter Data
filtered_data = np.zeros(shape3)

print('Filtering Data...')


# ~~~~~~~~~~~~~~Filtering~~~~~~~~~~~~~~~~~~~~~
for i in range(shape[0]):
  for j in range(shape[1]):
    # print(time_data[i,j])
    # temp = filter_data(time_data[i,j],.02,5,'low')
    filtered_data[i,j] = filter_data(time_data[i,j],.01,5,'high')
    # filtered_data[i,j] = temp
    # print('Filtered data:',filtered_data[i,j])
    
  if i % 20 == 0:
    print('Filter step',i,'of',shape[0])
    
    
print('Filtering Complete!')
    
    
    
#~~~~~~~~~~~~Processing~~~~~~~~~~~~~~~~~~~~~~~
for step in steps:

  if step >= max_steps:
    break
    
  #Open image, reset count
  count = 0
  
  
  
  #Current Image
  # cur = time_data[:,:,step]
  cur = filtered_data[:,:,step]
  
  #Go through each pixel
  for i,row in enumerate(cur):
    for j,col in enumerate(row):
      
      if step > 1:
        # print(cur[i,j])
        findChange[i,j] = findChange[i,j] + abs(int(cur[i,j]) - int(last[i,j]))
        #print(num,' FindChange',findChange[i,j])
      if col > threshold:
        count += 1
      
  # print('Find Change:\n',findChange)
  # pixelGraph.append(cur[300,300])
  counts.append(count)
  last = cur
  # print(len(counts))
  
  #Ever X frames, save image of the sum of changes
  if step % 10 == 0:
    print('Step:', step)
    # findChange = (findChange * 255/np.amax(findChange)).astype(np.uint8)
    # im = Image.fromarray(findChange)
    # im.save('Averaged/' + str(num) + 'Change.png',"PNG")
    # im.show()
    # findChange = np.zeros(shape)
    
    # break
  
    
    
time = range(len(counts))


#~~~~~~~~~~~~~~~~~~~~To Image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
findChange = (findChange * 255/np.amax(findChange)).astype(np.uint8)
print('FindChange:',findChange)
im = Image.fromarray(findChange)
im = im.convert('RGB')
# im.save('F1000_High.01.png',"PNG")
im.show()




#~~~~~~~~~~~~~~~~~~~~Map Neurons~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
key_in = 110

while(1):
  key_in = int(input('\nPick a light threshold(1-255, 0 to quit, -1 to save): '))
  
  if key_in == 0:
    break
  if key_in == -1:
    str_in = input('\nEnter filename(no filetype): ')
    im = Image.fromarray(neural_map)
    str_in += '.png'
    im = im.convert('RGB')
    im.save(str_in ,"PNG")
  neural_map = np.zeros(shape)
  
  for i in range(shape[0]):
    for j in range(shape[1]):
      if findChange[i][j] > key_in:
        neural_map[i][j] = 255
      
        
  im = Image.fromarray(neural_map)
  im.show()
  break
  
 

#~~~~~~~~~~~~~~~~~~Clustering~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Holding data -- row, column, value
cluster_data = np.zeros((1,2),dtype='int')

key_in = 10
while(1):
  # key_in = float(input('\nPick a clustering threshold(0 to quit): '))
  
  if key_in == 0:
    break
  
  c_threshold = key_in
  
  for i in range(shape[0]):
      for j in range(shape[1]):
        if neural_map[i,j] >0:
          cluster_data = np.vstack([cluster_data,[i,j]])
        

  clusters = hcluster.fclusterdata(cluster_data, c_threshold, criterion='distance')

  plt.scatter(*np.transpose(cluster_data), c=clusters)
  plt.axis("equal")
  title = "threshold: %f, number of clusters: %d" % (c_threshold, len(set(clusters)))
  plt.title(title)
  plt.show()
  break
  

#~~~~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spike_threshold = 15

spike_list = [[]]
used = []

hard_list = [(233,371),(242,371),(249,370),(244,379),(260,376)]

key_in = 1
while key_in != 0:
  key_in = int(input('\nPick Raster Plot Threshold: '))
  spike_threshold = key_in
  spike_list = [[]]
  
  #change to cluster_data
  for num,(i,j) in enumerate(hard_list):
    temp = []
    
    
    if True:
    # if clusters[num] not in used:
      #Loop through and append all spikes
      for s in range(max_steps):
        # print(time_data[i,j,s])
        # print('num',clusters[num])
        # print('used',used)
        if time_data[i,j,s] > spike_threshold:
          temp.append(s)
          
          # print('Did it')
        
      if len(temp) >= 1:
        spike_list.append(temp)
        used.append(clusters[num])
        
    
    
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
