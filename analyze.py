from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.misc import imshow
from scipy.misc import imresize

from scipy import signal


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
threshold = 180
scale = 4
max_steps = 1000


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
neural_map = np.zeros(shape)

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
  
  #im.show()
  
  #Create np array
  cur_temp = np.array(im)
  
  #Resizing
  cur = imresize(cur_temp,1/scale)
  
  
  
  time_data[:,:,num] = cur
  if num % 50 == 0:
    print('Step:', num)
  
    
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
    temp = filter_data(time_data[i,j],.01,5,'low')
    # filtered_data[i,j] = filter_data(time_data[i,j],.01,3,'high')
    filtered_data[i,j] = temp
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
im.save('F1000_Low.01.png',"PNG")
im.show()




#~~~~~~~~~~~~~~~~~~~~Map Neurons~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for i in range(shape[0]):
  for j in range(shape[1]):
    if findChange[i][j] > threshold:
      neural_map[i][j] = 255
      
im = Image.fromarray(neural_map)
im.show()



#~~~~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots()
# ax.plot(time,counts)

ax.plot(steps,time_data[0,0])

ax.set(xlabel='time',ylabel = 'magnitude',title = 'Activation above threshold vs. Time')
ax.grid()
#fig.savefig('ActivationFull.png')
plt.show()
