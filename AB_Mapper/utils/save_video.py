import cv2
import os
import argparse

def create_video(image_folder, video_name, frame_rate = 5):

	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	images.sort(key = lambda x: int(x[:-4]))

	frame = cv2.imread(os.path.join(image_folder, images[0]))
	size = frame.shape[1], frame.shape[0]

	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	vid = cv2.VideoWriter(video_name, fourcc, frame_rate, size, True)
	for image in images:

	    img = cv2.imread(os.path.join(image_folder, image))

	    vid.write(img)

	vid.release()
	print(video_name + " is created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default='saved_img/')
    parser.add_argument("--name", help="video name without extension", default="video")
    parser.add_argument("--fps", help="frame rate", type = int, default=5)
    args = parser.parse_args()
    create_video(args.dir, args.name+".avi", args.fps)

