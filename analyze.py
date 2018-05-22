from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.misc import imshow

files = Path('../Raw').glob('*.tif')

#1048 * 1328
threshold = 180
counts = []
pixelGraph = []
shape = (1048,1328)
last = np.zeros(shape)
findChange = np.zeros(shape)


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
  for i,row in enumerate(cur):
    for j,col in enumerate(row):
      if num > 1:
        findChange[i,j] = findChange[i,j] + abs(int(cur[i,j]) - int(last[i,j]))
        #print(num,' FindChange',findChange[i,j])
      if col > threshold:
        count += 1
      
  print('Find Change:\n',findChange)
  pixelGraph.append(cur[300,300])
  counts.append(count)
  last = cur
  print(len(counts))
  if num > 500:
    break
  
time = range(len(counts))

findChange = (findChange * 255/np.amax(findChange)).astype(np.uint8)
print('FindChange:',findChange)
im = Image.fromarray(findChange)
im.save('pixelChangeFull.png',"PNG")
im.show()


#Plotting
fig, ax = plt.subplots()
ax.plot(time,counts)

ax.set(xlabel='time',ylabel = 'magnitude',title = 'Activation above threshold vs. Time')
ax.grid()
fig.savefig('ActivationFull.png')
plt.show()
