
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

c.fft_display(409)
c.map_cells(110)



# c.simplify_frequencies(1.3,18000)
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
  c.fft_analysis()
  c.simplify_frequencies(thresh_list[0],thresh_list[1])
  
  #Do inverse
  key_in = input('Threshold for simplifying inv(amp, count):')
  key_in = key_in.replace(' ','')
  thresh_list = key_in.split(',')
  for num, thresh in enumerate(thresh_list):
    thresh_list[num] = float(thresh)
  
  c.simplify_frequencies_inv(thresh_list[0],thresh_list[1])
  


c.cluster_spacial(2)

c.plot_clusters(10,14,16,20,25)

c.make_avg_cluster_dict()

