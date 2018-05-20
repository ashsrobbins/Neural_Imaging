from PIL import Image
from pathlib import Path
import numpy



files = Path('../Raw').glob('*.tif')



for f in files:
  im = Image.open(f)
  #im.show()
  imarray = numpy.array(im)
  print(imarray)
  
  