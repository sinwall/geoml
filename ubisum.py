__version__ = '0.0.0'
# last update 2022/10/28

import os, sys

if sys.platform.startswith('linux'):
    if os.path.exists('../tools/google-cloud-sdk'):
        _here = 'colab'
    elif os.uname().nodename == 'lk4':
        _here = 'lk4'
elif sys.platform.startswith('win'):
    _here = 'pc'
else:
    raise NotImplementedError

if _here == 'pc':
    input_path = 'E:/database'
elif _here == 'lk4':
    input_path = '../input'
elif _here == 'colab':
    from google.colab import drive # noqa
    drive.mount('./drive')
    input_path = './drive/MyDrive/laboratory/input'
else:
    raise NotImplementedError