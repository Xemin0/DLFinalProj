import os
import sys
#sys.modules.pop('model.Generator')

from data.preprocess import TrainDatasetFromFolder
from model.Generator import Generator


#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import matplotlib.pyplot as plt

lr, hr = TrainDatasetFromFolder()
gen_model = Generator()

lr = (lr - 127.5) / 127.5
outputs = gen_model(lr[:2])

print(outputs.shape)

plt.imshow((outputs[0] + 1)/2)
plt.show()