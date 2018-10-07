import numpy as np
np.save("train_content", np.arange(0, 2*256*256*3).reshape([2, 128, 128, 12]).astype(np.float32))
np.save("train_style", np.arange(0, 2*256*256*3).reshape([2, 256, 256,3]).astype(np.float32))