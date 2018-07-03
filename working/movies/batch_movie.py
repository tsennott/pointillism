from pointillism import pointillize
import imageio
from PIL import Image
import time
from multiprocessing import Pool, freeze_support
from functools import partial
import numpy as np

def f(frames):
    im_out = []
    then = time.time()
    for j, frame in enumerate(frames):
        image = Image.fromarray(frame)
        point=pointillize(image=image, border=0, reduce_factor = 1, increase_factor = 2)
        point.plotRecPoints(40, multiplier=1, fill=True)
        point.plotRandomPointsComplexity(n=3e4, constant=0.005,
                                         power=2, locations=locations)
        im_out.append(np.array(point.out))
        now = time.time()
        if j%1 == 0: print('done frame %d of %d, elapsed time is %0.2f min' %
                                     (j,len(frames),(now-then)/60))
    return im_out
        

if __name__ == '__main__':

	reader = imageio.get_reader('movies/batch/F65C5430ABE1E4FB2FCB6AA435461BB4.mp4')
	fps = reader.get_meta_data()['fps']

	writer = imageio.get_writer('movies/56FC_full_out_mp.mp4', fps=fps)
	
	frames = []
	for i, im in enumerate(reader):
		frames.append(im)

	frames = frames[0:24]

	n_processes = 8
	chunk = len(frames)//n_processes
	frame_chunks = [frames[x:x+chunk] for x in range(0,len(frames),chunk)]

	image = Image.fromarray(frames[0])
	point=pointillize(image=image, border=0, reduce_factor = 1, increase_factor = 2)
	locations = point._generateRandomPoints(3e4)

	p = Pool(n_processes)

	start = time.time()
	out = p.map(f,frame_chunks)
	end = time.time()
	print('Took %0.2f minutes' % ((end-start)/60))

	out_frames = []
	for frames in out:
		for frame in frames:
		    out_frames.append(frame)

	for frame in out_frames:
		writer.append_data(frame[:,:,:])
	writer.close()


