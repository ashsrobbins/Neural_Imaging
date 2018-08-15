#RunCell.py

from Cell import Cell_Analysis
import os, glob
import numpy as np
from PIL import Image
import cv2

path = '../Raw'
files = glob.glob(os.path.join(path, '*.tif'))
files = sorted(files, key=lambda name: int(str(name[24:-15])))
scale = 4
max_steps = 3000



c = Cell_Analysis()
# c.data_from_path(path,scale,max_steps)
c.data_from_files(files,scale,max_steps)

c.fft_analysis()



# c.full_fft_avg()

# print('Begin writing:\n')
# for i in range(max_steps):
  
  # cur_name = 'Raw_Video/NUM_' + str(i) + '.png'
  # cv2.imwrite(cur_name, c.time_data[:,:,i].real)
  # if i% 250 == 0:
    # print('Step',i,'/',c.shape3[2])
# print('\nWriting Complete!\n')


# c.plot_fft_pixel((50,100))

#Remove specific frequencies
# for filt in filter_array:
  # c.fft_data[:,:,filt].fill(0)
  # c.fft_data[:,:,max_steps-filt].fill(0)
#Recreate time data_from_files
c.ifft_analysis()
c.full_fft_avg()
# c.data_to_img(c.recreated_data[:,:][:500].real,'After_Filtering/IM_')

c.plot_pixel(c.time_data,(50,100))
c.plot_pixel(c.recreated_data,(50,100))
c.show_plot()




#Choose fft value
freq = 0
while(1):
  key_in = int(input('FFT Value:(-1 to continue)'))
  if key_in == -1:
    break
  freq = key_in
  im = c.fft_display(key_in)
  im = im.convert('RGB')
  im.show()

while(1):
  key_in = int(input('Map threshold(0-255):(-1 to continue)'))
  if key_in == -1:
    break
  c.map_cells(key_in)
  
while(1):
  key_in = input('Threshold for simplifying(amp, count)(-1 to continue)(-2 to convert to video):')
  if key_in == '-1':
    break
  if key_in == '-2':
    print('Begin writing:\n')
    for i in range(max_steps):
      
      cur_name = 'Simplified_Freqs/NUM_' + str(i) + '.png'
      cv2.imwrite(cur_name, c.recreated_data[:,:,i].real)
      if i% 250 == 0:
        print('Step',i,'/',c.shape3[2])
    print('\nWriting Complete!\n')
    continue
    
  key_in = key_in.replace(' ','')
  thresh_list = key_in.split(',')
  for num, thresh in enumerate(thresh_list):
    thresh_list[num] = float(thresh)
  
  c.simplify_frequencies(thresh_list[0],thresh_list[1])
    
  
while(1):
  key_in = int(input('Cluster distance:(-1 to continue)'))
  if key_in == -1:
    break
    
  c.cluster_spacial(key_in)
  
  

c.make_avg_cluster_dict()

c.fft_cluster_analysis()

# c.cluster_fft_grad(4000)

c.cluster_fft(4000)
  
while(1):
  key_in = input('Input comma-separated Cell list:(-1 to continue)')
  if key_in == '-1':
    break
  
  key_in = key_in.replace(' ','')
  cell_list = key_in.split(',')
  for num, cell in enumerate(cell_list):
    cell_list[num] = int(cell)
  c.plot_clusters(cell_list)
