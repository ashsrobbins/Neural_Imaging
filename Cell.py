from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
import scipy
from scipy.misc import imshow
from scipy.misc import imresize
from scipy.fftpack import fft
import os, glob
from scipy import signal
import re
import scipy.cluster.hierarchy as hcluster
import cv2
import time




class Cell_Analysis(object):
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self):
    pass
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #Import .tif data from a directory pathlib
  #dir_path -----  str of path to directory, ex: '../DirectoryName'
  #scale    -----  size to downscale image, ex: 2
     
  #max_steps-----  maximum steps to process, leave blank to do all steps
  def data_from_path(self,dir_path:str, scale, max_steps = 0):
    files = list(Path('../Raw').glob('*.tif'))
    
    # files = sorted(files, key = lambda name: int(str(name)[24:-15]))
    #Get max steps and shape of data
    
    for num,f in enumerate(files):
      if num == 0:
        h,w = (np.array(Image.open(f))).shape
        self.shape = (int(h/scale),int(w/scale))
      self.max_steps = num + 1
      
    if max_steps > 0:
      self.max_steps = max_steps
    
    
    self.shape3 = (self.shape[0],self.shape[1],self.max_steps)
    self.time_data = np.zeros(self.shape3)
    print('H,W,T',self.shape3)
    
    
    files = Path('../Raw').glob('*.tif')
    
    for num,f in enumerate(files):
      if num >= max_steps:
        break
      #Open image, convert to array
      im = Image.open(f)
      cur_temp = np.array(im)
      
      #Resizing
      cur = imresize(cur_temp,1/scale)
      
      
      
      self.time_data[:,:,num] = cur
      if num % 50 == 0:
        print('Step:', num)
        
    print('Data Successfully Imported')
    
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #Import .tif data from a files
  #files    -----  sorted list of files
  #scale    -----  size to downscale image, ex: 2
     
  #max_steps-----  maximum steps to process, leave blank to do all steps
  def data_from_files(self,files, scale, max_steps = 0):
    
    #Get max steps and shape of data
    
    for num,f in enumerate(files):
      if num == 0:
        h,w = (np.array(Image.open(f))).shape
        self.shape = (int(h/scale),int(w/scale))
      self.max_steps = num + 1
      
    if max_steps > 0:
      self.max_steps = max_steps
    
    
    self.shape3 = (self.shape[0],self.shape[1],self.max_steps)
    self.time_data = np.zeros(self.shape3)
    print('H,W,T',self.shape3)
    
    ti = time.time()
    ti_tot = 0
    for num,f in enumerate(files):
      if num >= max_steps:
        break
      #Open image, convert to array
      # im = Image.open(f)
      # cur_temp = np.array(im)
      cur_temp = cv2.imread(f,-1)
      
      #Resizing
      cur = imresize(cur_temp,1/scale)
      
      
      
      self.time_data[:,:,num] = cur
      if num % 50 == 0:
        dt = time.time() - ti
        ti = time.time()
        ti_tot += dt
        print('Step:', num, 'Time:{0:.2f}'.format(dt))
        
    print('Data Successfully Imported\n Took {0:.2f} seconds'.format(ti_tot))

    
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #Pixel by pixel FFT analysis
  #Need to have self.time_data imported before calling
  def fft_analysis(self):
    self.fft_data = np.zeros(self.shape3, dtype = complex)
    
    print('Begin FFT Analysis...')
    
    #Loop through all data, take FFT
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        self.fft_data[i,j] = np.fft.fft(self.time_data[i,j]/255)
      
      if i % 20 == 0:
        print('FFT step',i,'of',self.shape[0])
        
    print('FFT Complete!')
        
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #Takes in a tuple(i,j) representing a pixel index
  #Plots frequency analysis of the pixel
  #Need to have self.fft_data to call
  def plot_fft_pixel(self, pixel):
    i,j = pixel
    n = self.fft_data[i,j].size
    
    delta_t = 1/120
    freqs = np.fft.fftfreq(n,delta_t)
    plt.plot(freqs[:n//2],self.fft_data[i,j][:n//2],'r')
    plt.axis([0,60,0,255])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #Takes in a tuple(i,j) representing a pixel index
  #Plots frequency analysis of the pixel
  #Need to have self.fft_data to call
  def plot_pixel(self,data, pixel):
    i,j = pixel
    n = data[i,j].size
    
    print('Length of Recreated:',n)
    plt.plot(range(n),data[i,j])
    plt.axis([0,n,0,255])
    plt.xlabel('Steps(1/120s)')
    plt.ylabel('Intensity')
    
  def show_plot(self):
    plt.show()
      
      
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def fft_display(self,fft_value = 136):
    #Average all frames values
    im = Image.fromarray(np.average(self.time_data,2))
    # im = Image.fromarray(self.time_data[:,:,0])
    im = im.convert('RGB')
    pix = im.load()
    self.fft_freq = fft_value
    
    mag = np.absolute(self.fft_data[:,:,fft_value])
    norm_mag = mag/np.amax(mag)
    
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        # pix[j,i] = (norm_mag[i,j]*pix[j,i][0],pix[j,i][1],pix[j,i][2])
        pix[j,i] = (int(norm_mag[i,j]*255),pix[j,i][1],pix[j,i][2])
      
    # im.show()
    self.cell_im = im
    return im
    
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #Create a map of cells that have a light value over a certain threshold
  #Needs to have fft_display called first to get a value to generate a map of
  def map_cells(self, threshold):
    self.neural_map = np.zeros(self.shape)
    self.light_threshold = threshold
    pix = self.cell_im.load()
    
    
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        if pix[j,i][0] > threshold:
          self.neural_map[i][j] = 255
          
    map = Image.fromarray(self.neural_map)
    map.show()
      
      
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #
  #Cluster data within a spacial distance
  #Creates self.cluster_data
  def cluster_spacial(self, distance):
    self.cluster_data = np.zeros((1,2),dtype='int')
    self.centroid_dict = dict()
    diff_clusters = None
    pix = self.cell_im.load()
    
    
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        if pix[j,i][0] > self.light_threshold:
          self.cluster_data = np.vstack([self.cluster_data,[j,i]])
    
    
    print('Total Above Threshold =',len(self.cluster_data))
    
    input('Cluster? Press enter:')
    print('Clustering...')
    self.clusters = hcluster.fclusterdata(self.cluster_data, distance, criterion='distance')
    
    
    #Plot data
    plt.scatter(*np.transpose(self.cluster_data), c=self.clusters)
    diff_clusters = set(self.clusters)
    self.centroid_dict = dict.fromkeys(diff_clusters,(0,0))
    
    #Iterate through groups, average indices
    for group in diff_clusters:
      count = 0
      
      for num,(i,j) in enumerate(self.cluster_data):
        if self.clusters[num] == group:
          self.centroid_dict[group] = (self.centroid_dict[group][0] + i, self.centroid_dict[group][1] + j)
          count += 1
      
      #Average and annotate plot
      self.centroid_dict[group] = tuple(cur/count for cur in self.centroid_dict[group])
      plt.annotate(str(group),xy=self.centroid_dict[group])
  
  
  
  
  
    #plt.annotate(clusters,
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (distance, len(set(self.clusters)))
    plt.title(title)
    # plt.imshow(neural_map)
    plt.imshow(np.average(self.time_data,2))
    plt.show()
    
    
    
    
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #Plots numbered clusters
  #Takes in list of ints representing cluster groups, plottign those values
  #through time
  #Needs self.clusters(call cluster_spacial
  #Also plots the frequency from the frequency analysis
  def plot_clusters(self, cell_list):
    diff_clusters = set(self.clusters)
    avg_dict = dict()
    group_avg = 0
    phase_avg = 0
    phase_count = 0
    
    print('Plotting chosen clusters...')
  
    
    #Iterate through each cluster, averaging whole cluster over time
    chosen_clusters = (cl for cl in diff_clusters if cl in cell_list)
    for group in chosen_clusters:
      #Add list for each cluster group
      avg_dict[group]= []
      for t in range(self.max_steps):
        
        count = 0
        group_avg = 0
        for num,(j,i) in enumerate(self.cluster_data):
          if self.clusters[num] == group:
            # group_avg += self.time_data[i,j,t]
            group_avg += self.recreated_data[i,j,t]
            
            
            count += 1
        
        #Append avg
        avg_dict[group].append(group_avg/count)
        
      #Plot current cluster  
      plt.plot(range(self.max_steps),avg_dict[group],label=str(group))
    
    #Plot frequency
    
    fs = 120#sampling rate
    time = np.arange(0,self.max_steps,1)
    ff = self.fft_freq#/120#frequency of signal
    
    for num,(j,i) in enumerate(self.cluster_data):
      phase_avg += self.fft_data[i,j,self.fft_freq]
      phase_count = num + 1
    phase_avg /= phase_count
    
    y = 75 + 10*np.sin(2*np.pi*ff/fs*time/self.max_steps )
    # y = 75 + 10*np.sin((2*np.pi+ np.arctan(phase_avg.real,phase_avg.imag))*ff/fs*time/self.max_steps )
    
    plt.plot(time,y)
    plt.legend()
    plt.show()
    
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #Averages each cluster, getting time data for the clusters
  #saves the dict to be accessible
  #Needs cluster_spacial to be called first
  def make_avg_cluster_dict(self):

    self.avg_cluster_dict = dict()
    group_avg = 0
    
    
    #Iterate through each cluster, averaging whole cluster over time
    chosen_clusters = (cl for cl in set(self.clusters))
    for q,group in enumerate(chosen_clusters):
      #Add list for each cluster group
      self.avg_cluster_dict[group] = []
      for t in range(self.max_steps):
        
        count = 0
        group_avg = 0
        for num,(j,i) in enumerate(self.cluster_data):
          if self.clusters[num] == group:
            # group_avg += self.time_data[i,j,t]
            group_avg += self.recreated_data[i,j,t]
            
            
            count += 1
        
        
        #Append avg
        self.avg_cluster_dict[group].append(group_avg/count)
        
      if q % 5 == 0:
        print('Step', q)
        
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #Does cluster analysis on the average cluster dict
  #Used for clustering the clusters
  def fft_cluster_analysis(self):
    self.fft_avg_cluster_data = np.zeros(self.shape3, dtype = complex)
    self.avg_cluster_fft_dict = dict()
    print('Begin Cluster FFT Analysis...')
    
    clusters = (cl for cl in set(self.clusters))
    
    for i,group in enumerate(clusters):
      self.avg_cluster_fft_dict[group] = np.fft.fft(self.avg_cluster_dict[group])
      
      if i % 10 == 0:
        print('Step',i,'/')# + str(len(list(clusters))))
        
    print('Cluster FFT Complete!')
    
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #Cluster the FFT data of the clusters to determine clusters firing together
  #Needs fft_cluster_analysis to be called first
  def cluster_fft(self,distance):
  
    #make list of magnitudes(instead of freqs)
    to_cluster = np.absolute(list(self.avg_cluster_fft_dict.values()))
    
    #Cluster  
    self.clusters2 = hcluster.fclusterdata(to_cluster, distance, criterion='distance')
    
    print('Clustered Clusters:\n',self.clusters2)
     
    #Plot data
    # plt.scatter(*np.transpose(self.cluster_data), c=self.clusters)
    plt.scatter(*np.transpose(list(self.centroid_dict.values())), c=self.clusters2)
    diff_clusters = set(self.clusters)
    # centroid_dict = dict.fromkeys(diff_clusters,(0,0))
    
    #Iterate through groups, average indices
    for num,group in enumerate(diff_clusters):
      
      
      #Plot new cluster number with old centroid
      tag = str(group) +'(' +  str(self.clusters2[num]) + ')'
      plt.annotate(tag,xy=self.centroid_dict[group])
  
  
  
  
  
    #plt.annotate(clusters,
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (distance, len(set(self.clusters2)))
    plt.title(title)
    # plt.imshow(neural_map)
    plt.imshow(np.average(self.time_data,2))
    plt.show() 
     
     
     
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #Cluster the FFT data of the clusters to determine clusters firing together
  #Needs fft_cluster_analysis to be called first
  def cluster_fft_grad(self,distance):
  
    #make list of magnitudes(instead of freqs)
    to_cluster = np.gradient(np.absolute(list(self.avg_cluster_fft_dict.values())))
    
    #Cluster  
    self.clusters2 = hcluster.fclusterdata(to_cluster, distance, criterion='distance')
    
    print('Clustered Clusters:\n',self.clusters2)
     
    #Plot data
    # plt.scatter(*np.transpose(self.cluster_data), c=self.clusters)
    plt.scatter(*np.transpose(list(self.centroid_dict.values())), c=self.clusters2)
    diff_clusters = set(self.clusters)
    # centroid_dict = dict.fromkeys(diff_clusters,(0,0))
    
    #Iterate through groups, average indices
    for num,group in enumerate(diff_clusters):
      
      
      #Plot new cluster number with old centroid
      tag = str(group) +'(' +  str(self.clusters2[num]) + ')'
      plt.annotate(tag,xy=self.centroid_dict[group])
  
  
  
  
  
    #plt.annotate(clusters,
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (distance, len(set(self.clusters2)))
    plt.title(title)
    # plt.imshow(neural_map)
    plt.imshow(np.average(self.time_data,2))
    plt.show() 
     
     
        
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  #Averages FFT throughout all time, plotting value
  def full_fft_avg(self):
    n = self.fft_data[0,0].size
    
    full_fft_data = np.average(self.fft_data, (0,1))
    delta_t = 1/120
    freqs = np.fft.fftfreq(n,delta_t)
    
    #Plot average Time data`
    plt.figure(1)
    plt.subplot(311)
    plt.title('Mean time Analysis')
    plt.xlabel('Time(s)')
    plt.ylabel('Intensity')
    plt.plot(np.arange(self.max_steps),np.average(self.time_data,(0,1)))
    
    
    
    
    #Plot average Frequency Data
    plt.subplot(312)
    plt.scatter(freqs[:n//2],np.absolute(full_fft_data[:n//2]))
    plt.axis([0,60,0,1])
    plt.title('Mean Frequency Analysis')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    
    
    #Plot recreated time data
    recreated = np.fft.ifft(full_fft_data)
    plt.subplot(313)
    # plt.plot(np.arange(n),recreated.real*255)
    
    w = scipy.fftpack.rfft(np.average(self.time_data,(0,1)))
    f = scipy.fftpack.rfftfreq(self.max_steps,1)
    
    spectrum = w**2
    
    cutoff_idx = spectrum < (spectrum.max()/50000)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    
    y2 = scipy.fftpack.irfft(w2)
    

    plt.plot(np.arange(self.max_steps),np.average(self.time_data,(0,1)),'r')
    plt.plot(np.arange(n),y2,'b')
    
    plt.show()
      
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  def data_to_img(self, data_array, path_name):
    shape = data_array.shape
    for num in range(shape[2]):
      im = Image.fromarray(data_array[:,:,num])
      cur_name = path_name + str(num) + '.png'
      im = im.convert('RGB')
      # im.show()
      im.save(cur_name, "PNG")
      
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
  def ifft_analysis(self):
    self.recreated_data = np.zeros(self.shape3, dtype = complex)
    
    print('Begin IFFT Recreation...')
    
    #Loop through all data, take FFT
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        self.recreated_data[i,j] = np.fft.ifft(self.fft_data[i,j])*255
      
      if i % 20 == 0:
        print('IFFT step',i,'of',self.shape[0])
        
    print('IFFT Complete!')
      
      
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  #Remove certain frequencies, keep interesting ones
  #   Interesting frequencies include those which most pixels are NOT oscillating
  #amp_threshold : int, value to check if amplitude at specific frequency is above
  #count_threshold: int, check if this many pixels are above the specified threshold
  #Otherwise remove these frequencies(to get fft_data back, fft_analysis must be called again0
  def simplify_frequencies(self, amp_thresh, count_thresh):
  
    
    # self.fft_analysis()
    #For each frequency:
    #   Generate magnitude array
    for i in range(self.shape3[2]):
      mags = np.absolute(self.fft_data[:,:,i])
      vals_above = (mags > amp_thresh).sum()
      #Counts below the threshold are wiped
      if vals_above < count_thresh:
        # print('Vals Above:', vals_above,'\ti:',i)
        self.fft_data[:,:,i].fill(0)
      if i%500 == 0:
        print('Step',i,'/',self.shape3[2])
        
    #Plot before data averaged
    plt.figure(1)
    plt.title('Mean time Analysis before')
    plt.xlabel('Time(s)')
    plt.ylabel('Intensity')
    plt.plot(np.arange(self.max_steps),np.average(self.time_data,(0,1)),'b')
    
    self.ifft_analysis()
    
    #Plot simplified data averaged
    plt.plot(np.arange(self.max_steps),np.average(self.recreated_data,(0,1)),'r')
    
    plt.show()
    
  def simplify_frequencies_inv(self, amp_thresh, count_thresh):
  
    
    # self.fft_analysis()
    #For each frequency:
    #   Generate magnitude array
    for i in range(self.shape3[2]):
      if i == 0:
        continue
      mags = np.absolute(self.fft_data[:,:,i])
      vals_above = (mags < amp_thresh).sum()
      #Counts below the threshold are wiped
      if vals_above < count_thresh:
        # print('Vals Above:', vals_above,'\ti:',i)
        self.fft_data[:,:,i].fill(0)
      if i%500 == 0:
        print('Step',i,'/',self.shape3[2])
        
    #Plot before data averaged
    plt.figure(1)
    plt.title('Mean time Analysis before')
    plt.xlabel('Time(s)')
    plt.ylabel('Intensity')
    plt.plot(np.arange(self.max_steps),np.average(self.time_data,(0,1)),'b')
    
    self.ifft_analysis()
    
    #Plot simplified data averaged
    plt.plot(np.arange(self.max_steps),np.average(self.recreated_data,(0,1)),'r')
    
    plt.show()
        
        
        
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  #Saves a dictionary to a file.
  
  
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  #Animates a graph of time data at 120fps, meant to be used for clustered data
  #Saves the file with the filename, adding 'mp4'
  def animate_graph(self,data,filename):
    def update_line(num,data,line):
      line.set_data(data[...,:num])
      return line
      
    f = filename + '.mp4'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=120,metadata=dict(artist='Ash Robbins'),bitrate=1800)
    
    fig1 = plt.figure()
    l, = plt.plot([],[],'r-')
    
    plt.xlim(0,self.max_steps)
    plt.ylim(0,255)
    
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Clustered Firing')
    
    #Animate and save
    line_ani = animation.FuncAnimation(fig1,update_line,self.max_steps,fargs=(data,l))
    line_ani.save(f,writer=writer)
    
    
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
  #Animates a graph of time data at 120fps, meant to be used for clustered data
  #Saves the file with the filename, adding 'mp4'
  def animate_graph_mult(self,data,filename):
    def update_line(num,data,lines):
      for i,line in enumerate(lines):
        if i < len(lines) - 1:
          line.set_data(data[0][:num],data[i + 1,:num])
      return lines
      
    f = filename + '.mp4'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=120,metadata=dict(artist='Ash Robbins'),bitrate=1800)
    
    fig1 = plt.figure()
    ls = []
    
    #Might be len -1
    for i in range(len(data)):
      ls+=plt.plot([],[])
    
    plt.xlim(0,self.max_steps)
    plt.ylim(0,255)
    
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Clustered Firing')
    
    #Animate and save
    line_ani = animation.FuncAnimation(fig1,update_line,self.max_steps,fargs=(data,ls))
    line_ani.save(f,writer=writer)
    
    
  
  #Saves loaded data to specified filename
  def save_data(self, filename, data = None):
    if data == None:
      data = self.time_data
    
    np.savez(filename,data = data)
    print('Data successfully saved at',filename + '.npz')

  def load_data(self, filename):
    npz = np.load(filename + '.npz')
    self.time_data = npz[npz.files[0]]
    _,_,self.max_steps = self.time_data.shape
    print('Data successfully Loaded')
    
    
  def graph_to_video(self, im_array,path):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
    fig1 = plt.figure()
    
    
    