from Cell import Cell_Analysis
import os, glob
import numpy as np
from PIL import Image
import cv2

path = '../Raw_Control'
files = glob.glob(os.path.join(path, '*.tif'))
files = sorted(files, key=lambda name: int(str(name[33:-15])))
scale = 4
max_steps = 2600



c = Cell_Analysis()
# c.data_from_path(path,scale,max_steps)
c.data_from_files(files,scale,max_steps)

c.fft_analysis()


c.fft_display(24)
c.map_cells(150)
c.simplify_frequencies(1.3,18000)
c.cluster_spacial(2)

c.make_avg_cluster_dict()

x = np.array([np.arange(c.max_steps),c.avg_cluster_dict[20],c.avg_cluster_dict[21],c.avg_cluster_dict[22],c.avg_cluster_dict[23])