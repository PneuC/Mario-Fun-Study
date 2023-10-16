import glob
import struct
import json

if __name__ == '__main__':
    for fpath in glob.glob('./*.txt'):
        with open(fpath, 'r') as f:
            content = f.read()
        newfpath = fpath[:-6] + '.rep'
        actlist = [int(c[1:-1]) for c in content[1:-1].split(', ')]
        bcontent = b''.join(struct.pack('B', a) for a in actlist)
        with open(newfpath, 'wb') as f:
            f.write(bcontent)
    # with open(newfpath, 'wb') as f:
    #     f.write(bcontent)
    pass

