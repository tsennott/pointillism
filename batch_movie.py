from pointillism import pointillize
import imageio
from PIL import Image
import time
from multiprocessing import Pool, freeze_support
from functools import partial
import numpy as np

def f(frames, n_processes, locations, i):
        length=len(frames)
        im_out = []
        start = int(i*length/n_processes)
        end = int((i+1)*length/n_processes)
        for i in range(start, end):
            image = Image.fromarray(frames[i])
            point=pointillize(image=image, border=0)
            point.plotRecPoints(40, multiplier=1, fill=True)
            point.plotRandomPointsComplexity(n=3e4, constant=0.0075,
                                             power=1.5, locations=locations)
            im_out.append(np.array(point.out))
            print('done frame %d of %d' %(i,length))
        return im_out, [start, end]

if __name__ == '__main__':
    reader = imageio.get_reader('movies/Cesaz Chavez 20171014 FAST.mp4')
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer('movies/cc_short_out_mp2.mp4', fps=fps)

    frames = []
    for i, im in enumerate(reader):
        frames.append(im)

    image = Image.fromarray(frames[0])
    point=pointillize(image=image, border=0)
    locations = point._generateRandomPoints(3e4)
    print('Made initial location points...')

    n_processes = 1
    f_with_args = partial(f, frames, n_processes, locations)

    p = Pool(4)
    start = time.time()
    out = p.map(f_with_args,range(0,4))
    end = time.time()
    print('DONE. Took %0.2f minutes' % ((end-start)/60))
    
    out_frames = []
    for frames in out:
        for frame in frames[0]:
            out_frames.append(frame)

    for frame in out_frames:
        writer.append_data(frame[:,:,:])
    writer.close()
    p.close()
    print('Saving done. Closing...')