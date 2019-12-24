import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Load train andd test videos from file:
input_location = 'data'
#full_datacap = cv.VideoCapture(os.path.join(input_location, 'train.mp4'))
test = cv.VideoCapture(os.path.join(input_location, 'test.mp4'))
output_location  = 'data/images'
#Read speed file
file_name =  os.path.join(input_location, 'train.txt')
training_speed = np.genfromtxt(file_name)
#This notebook is based on jovsa's github
#https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames


# Extract frames from train data and write data to csv
def process_data(input_path, output_path, train_speed, dataset_type,):
	print("Start")
	cap = cv.VideoCapture(os.path.join(input_path, dataset_type+'.mp4'))
	# create empty dictionary 
	full_data_dict = {}
	# Get number of frames
	video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	assert(len(train_speed)==video_length)
	count = 0
	time_start = time.time()
	# Start converting the video
	while cap.isOpened():
		# Extract the frame
		ret, frame = cap.read()
        # Write the results back to output location.
		img_path = output_path + '/%#05d.jpg' % (count)
		cv.imwrite(output_path + '/%#05d.jpg' % (count), frame)
		speed = float('NaN') if dataset_type == 'test' else train_speed[count]
		full_data_dict[count] = [img_path, count, speed]
		count = count + 1
		# If there are no more frames left
		if (count == video_length):
			# Release the feed
			cap.release()
			#break
			full_data = pd.DataFrame.from_dict(full_data_dict, orient='index')
			full_data.columns = ['image_path', 'image_index', 'speed']
			full_data.to_csv(output_path + dataset_type+'.csv', index=False)
	print("Finish")
	#print(time_end - time_start)

if __name__=="__main__":
	process_data(input_location, output_location, training_speed, 'train')
	fig, ax = plt.subplots(figsize=(20,10))
	#plt.plot(full_data['speed'])
	#plt.xlabel('image_index (or time since start)')
	#plt.ylabel('speed')
	#plt.title('Speed vs time')
	#plt.show()
