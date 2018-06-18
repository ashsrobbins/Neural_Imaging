from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.misc import imshow

from scipy import signal


files = Path('../Raw').glob('*.tif')



def filter_data():
  


#1048 * 1328
threshold = 180
counts = []
pixelGraph = []
shape = (1048,1328)
shape3 = (1048,1328,3072)

last = np.zeros(shape)
findChange = np.zeros(shape)
time_data = np.zeros(shape3)

pix_time = [[[]]]


#im = Image.fromarray(findChange)
#im.show()


for num,f in enumerate(files):
  #Open image, reset count
  im = Image.open(f)#,dtype='int64')
  count = 0
  
  
  
  #im.show()
  
  #Create np array
  cur = np.array(im)
  #print('MAX',np.amax(cur))
  
  time_data[:,:,num] = cur
  # pix_time[0][0].append(cur[0,0])
  # print("Array:",time_data[0,0,:15])
  # print('Current:',cur[0,0])
  # for i,row in enumerate(cur):
    # for j,col in enumerate(row):
      
      # if num > 1:
        # findChange[i,j] = findChange[i,j] + abs(int(cur[i,j]) - int(last[i,j]))
        # print(num,' FindChange',findChange[i,j])
      # if col > threshold:
        # count += 1
      
  # print('Find Change:\n',findChange)
  pixelGraph.append(cur[300,300])
  counts.append(count)
  last = cur
  # print(len(counts))
  
  #Ever X frames, save image of the sum of changes
  if num % 50 == 0:
    print('Step:', num)
    # findChange = (findChange * 255/np.amax(findChange)).astype(np.uint8)
    # im = Image.fromarray(findChange)
    # im.save('Averaged/' + str(num) + 'Change.png',"PNG")
    # im.show()
    # findChange = np.zeros(shape)
    
    # break
  if num > 1000:
    break
time = range(len(counts))



findChange = (findChange * 255/np.amax(findChange)).astype(np.uint8)
print('FindChange:',findChange)
im = Image.fromarray(findChange)
#im.save('pixelChange.png',"PNG")
#im.show()


#Plotting
fig, ax = plt.subplots()
# ax.plot(time,counts)
steps = range(len(time_data[0,0]))
ax.plot(steps,time_data[0,0])

ax.set(xlabel='time',ylabel = 'magnitude',title = 'Activation above threshold vs. Time')
ax.grid()
#fig.savefig('ActivationFull.png')
plt.show()
