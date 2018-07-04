#Author: Maria Rowe R. Riomalos 	2014-05889
#Reference: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

# import the necessary packages
import datetime
import imutils
import time
import cv2
import sys
import numpy as np

from random import randint
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi

#initialize variables
pts = {}			#pts for containing all the points each contour goes to
results = {}		#contains all the path each contour traveled
pestColorsDict = {}	#contains the assigned colors for each contour
pestColors = []		#contains all the random colors generated from colorGenerator
countVisitsPerRegion = [0,0,0,0,0,0,0,0]	#counts the number of visits per region

def colorGenerator():
	B = randint(0, 255)
	G = randint(0, 255)
	R = randint(0, 255)
	return [B,G,R]

for i in range(0,100):
	#generate color for pest
	tempColor = colorGenerator()
	if tempColor != [255,0,0]:
		pestColors.append(tempColor)

class InsectTracker(QDialog):
	def __init__(self):
		super(InsectTracker,self).__init__()
		loadUi('InsectTracker.ui',self)
		self.image = None
		self.firstFrame = None
		self.openLaptopCameraButton.clicked.connect(self.setup_laptop_camera)
		self.openExtendedCameraButton.clicked.connect(self.setup_extended_camera)
		self.stopButton.clicked.connect(self.stop_timer)
		self.startTrackingButton.clicked.connect(self.start_track_pests)

	#read from webcam

	def setup_laptop_camera(self):
		self.camera = cv2.VideoCapture(0)
	 	self.openExtendedCameraButton.setEnabled(False)
    	# QTimer.singleShot(5000, lambda: self.targetBtn.setDisabled(False))
		self.start_webcam()

	def setup_extended_camera(self):
		self.camera = cv2.VideoCapture(1)
		self.openLaptopCameraButton.setEnabled(False)
		self.start_webcam()

	def start_webcam(self):
		self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
		self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)

		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update_frame)
		self.timer.start(5) #video will appear after 5 msec

	def update_frame(self):
		ret, self.image = self.camera.read()
		self.image = cv2.flip(self.image,1)
		# put regions on the video
		cv2.line(self.image, (self.image.shape[1]/3, 0), (self.image.shape[1]/3, self.image.shape[0]), (255, 0, 0), 1, 1)
		cv2.line(self.image, (2*(self.image.shape[1]/3), 0), (2*(self.image.shape[1]/3), self.image.shape[0]), (255, 0, 0), 1, 1)
		cv2.line(self.image, (0, self.image.shape[0]/3), (self.image.shape[1], self.image.shape[0]/3), (255, 0, 0), 1, 1)
		cv2.line(self.image, (0, 2*(self.image.shape[0]/3)), (self.image.shape[1], 2*(self.image.shape[0]/3)), (255, 0, 0), 1, 1)

		cv2.putText(self.image, "Region 1", (self.image.shape[1]/3 - 70, self.image.shape[0]/3 - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 2", (2*(self.image.shape[1]/3) - 70, self.image.shape[0]/3 - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 3", (self.image.shape[1] - 70, self.image.shape[0]/3 - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 4", (self.image.shape[1]/3 - 70, 2*(self.image.shape[0]/3) - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 5", (self.image.shape[1] - 70, 2*(self.image.shape[0]/3) - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 6", (self.image.shape[1]/3 - 70, self.image.shape[0] - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 7", (2*(self.image.shape[1]/3) - 70, self.image.shape[0] - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(self.image, "Region 8", (self.image.shape[1] - 70, self.image.shape[0] - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		self.displayImage(self.image,1)

	def stop_timer(self):
		self.timer.stop()

	def start_track_pests(self):
		self.time_left_int = int(self.durationInput.text()) * 4
		self.resultingImage = self.image
		self.timer.timeout.connect(self.track_pests)
		self.timer.start(0) #video will appear after 5 msec
		self.update_time()

	def track_pests(self):
		self.time_left_int -= 1

		if self.time_left_int > 0:
			ret, self.image = self.camera.read()
			self.image = cv2.flip(self.image,1)

			# put regions on the video
			cv2.line(self.image, (self.image.shape[1]/3, 0), (self.image.shape[1]/3, self.image.shape[0]), (255, 0, 0), 1, 1)
			cv2.line(self.image, (2*(self.image.shape[1]/3), 0), (2*(self.image.shape[1]/3), self.image.shape[0]), (255, 0, 0), 1, 1)
			cv2.line(self.image, (0, self.image.shape[0]/3), (self.image.shape[1], self.image.shape[0]/3), (255, 0, 0), 1, 1)
			cv2.line(self.image, (0, 2*(self.image.shape[0]/3)), (self.image.shape[1], 2*(self.image.shape[0]/3)), (255, 0, 0), 1, 1)

			cv2.putText(self.image, "Region 1", (self.image.shape[1]/3 - 70, self.image.shape[0]/3 - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 2", (2*(self.image.shape[1]/3) - 70, self.image.shape[0]/3 - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 3", (self.image.shape[1] - 70, self.image.shape[0]/3 - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 4", (self.image.shape[1]/3 - 70, 2*(self.image.shape[0]/3) - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 5", (self.image.shape[1] - 70, 2*(self.image.shape[0]/3) - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 6", (self.image.shape[1]/3 - 70, self.image.shape[0] - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 7", (2*(self.image.shape[1]/3) - 70, self.image.shape[0] - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.putText(self.image, "Region 8", (self.image.shape[1] - 70, self.image.shape[0] - 05), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

			# preprocessing
			# resize the frame, convert it to grayscale, and blur it
			self.image = imutils.resize(self.image, width=900)
			gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (21, 21), 0)

			# if the first frame is None, initialize it
			if self.firstFrame is None:
				self.firstFrame = gray
				self.track_pests()
			# segmentation
			# compute the absolute difference between the current frame and first frame
			frameDelta = cv2.absdiff(self.firstFrame, gray)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

			# dilate the thresholded image to fill in holes, then find contours on thresholded image
			thresh = cv2.dilate(thresh, None, iterations=2)
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			countPest = 0

			# loop over the contours
			for c in cnts:
				# if the contour is too small, ignore it
				if cv2.contourArea(c) > 1000:

					# compute the center of the contour
					M = cv2.moments(c)
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
					center = (cX,cY)
					countPest = countPest + 1		#update countPest
					idName = "Pest" + str(countPest)


					if (idName not in pts.keys()):
						pts[idName] = []
						pestColorsDict[idName] = pestColors[countPest]
					else:
						pts[idName].append(center)

					if (idName not in results.keys()):
						results[idName] = []

					# check if center has entered an eggplant region
					if cX < self.image.shape[1]/3 and cY < self.image.shape[0]/3:
						if len(results[idName]) == 0:
							results[idName].append("Region 1")
							countVisitsPerRegion[0] = countVisitsPerRegion[0] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 1":
							results[idName].append("Region 1")
							countVisitsPerRegion[0] = countVisitsPerRegion[0] + 1
					elif (cX > self.image.shape[1]/3 and cX < 2*(self.image.shape[1]/3)) and cY < self.image.shape[0]/3:
						if len(results[idName]) == 0:
							results[idName].append("Region 2")
							countVisitsPerRegion[1] = countVisitsPerRegion[1] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 2":
							results[idName].append("Region 2")
							countVisitsPerRegion[1] = countVisitsPerRegion[1] + 1
					elif cX > 2*(self.image.shape[1]/3) and cY < self.image.shape[0]/3:
						if len(results[idName]) == 0:
							results[idName].append("Region 3")
							countVisitsPerRegion[2] = countVisitsPerRegion[2] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 3":
							results[idName].append("Region 3")
							countVisitsPerRegion[2] = countVisitsPerRegion[2] + 1
					elif cX < self.image.shape[1]/3 and (cY > self.image.shape[0]/3 and cY < 2*(self.image.shape[0]/3)):
						if len(results[idName]) == 0:
							results[idName].append("Region 4")
							countVisitsPerRegion[3] = countVisitsPerRegion[3] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 4":
							results[idName].append("Region 4")
							countVisitsPerRegion[3] = countVisitsPerRegion[3] + 1
					elif cX > self.image.shape[1]/3 and (cY > self.image.shape[0]/3 and cY < 2*(self.image.shape[0]/3)):
						if len(results[idName]) == 0:
							results[idName].append("Region 5")
							countVisitsPerRegion[4] = countVisitsPerRegion[4] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 5":
							results[idName].append("Region 5")
							countVisitsPerRegion[4] = countVisitsPerRegion[4] + 1
					elif cX < self.image.shape[1]/3 and cY > 2*(self.image.shape[0]/3):
						if len(results[idName]) == 0:
							results[idName].append("Region 6")
							countVisitsPerRegion[5] = countVisitsPerRegion[5] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 6":
							results[idName].append("Region 6")
							countVisitsPerRegion[5] = countVisitsPerRegion[5] + 1
					elif (cX > self.image.shape[1]/3 and cX < 2*(self.image.shape[1]/3)) and cY > 2*(self.image.shape[0]/3):
						if len(results[idName]) == 0:
							results[idName].append("Region 7")
							countVisitsPerRegion[6] = countVisitsPerRegion[6] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 7":
							results[idName].append("Region 7")
							countVisitsPerRegion[6] = countVisitsPerRegion[6] + 1
					elif cX > 2*(self.image.shape[1]/3) and cY > 2*(self.image.shape[0]/3):
						if len(results[idName]) == 0:
							results[idName].append("Region 8")
							countVisitsPerRegion[7] = countVisitsPerRegion[7] + 1
						elif len(results[idName]) > 0 and results[idName][len(results[idName]) - 1] != "Region 8":
							results[idName].append("Region 8")
							countVisitsPerRegion[7] = countVisitsPerRegion[7] + 1

					centroid = idName + " (" + str(cX) + ", " + str(cY) + ")"

					cv2.circle(self.image, center, 3, (pestColorsDict[idName][0], pestColorsDict[idName][1], pestColorsDict[idName][2]), 1)
					cv2.putText(self.image, centroid, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (pestColorsDict[idName][0], pestColorsDict[idName][1], pestColorsDict[idName][2]), 2)
					# cv2.putText(self.image, "Time elapsed: " + str(round(elapsed,2)) + " seconds", (self.image.shape[1]/3 + 40, 2*(self.image.shape[0]/3) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

					# compute the bounding box for the contour, draw it on the frame, and update the text
					(x, y, w, h) = cv2.boundingRect(c)
					cv2.rectangle(self.image, (x, y), (x + w, y + h), (pestColorsDict[idName][0], pestColorsDict[idName][1], pestColorsDict[idName][2]), 2)

					# draw the path
					for k, v in pts.items():
						for i in xrange(1,len(v)):
							cv2.line(self.image, pts[k][i - 1], pts[k][i], (pestColorsDict[k][0], pestColorsDict[k][1], pestColorsDict[k][2]), 2)
				self.resultingImage = self.image
				self.displayImage(self.image,1)
		else:
			self.timer.stop()
			cv2.imwrite("result.jpg", self.resultingImage)
			resultFile = open("result.txt","w")
			for k, v in results.items():
				resultFile.write(k + ": " + str(v) +"\n")
			resultFile.close()
			# report the best pest-resistant region and the least pest-resistant region
			minIndex = countVisitsPerRegion.index(min(countVisitsPerRegion))
			maxIndex = countVisitsPerRegion.index(max(countVisitsPerRegion))
			self.resultLabel.setText("Most visited region is Region " + str(maxIndex + 1) + " and the least visited region is " + str(minIndex + 1) + ".")
		self.update_time()

	def update_time(self):
		self.timeLabel.setText(str(self.time_left_int/4) + " seconds")

	def displayImage(self, img, window = 1):
		qformat = QImage.Format_Indexed8
		if len(img.shape) == 3:	#[0]=rows, [1]=cols, [2]=channels
			if (img.shape[2]) == 4:
				qformat = QImage.Format_RGBA8888
			else:
				qformat = QImage.Format_RGB888
		outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
		#BGR>>RGB
		outImage = outImage.rgbSwapped()

		if window == 1:
			self.videoLabel.setPixmap(QPixmap.fromImage(outImage))
			self.videoLabel.setScaledContents(True)

if __name__=='__main__':
	app = QApplication(sys.argv)
	window = InsectTracker()
	window.setWindowTitle('Pest Tracker')
	window.show()
	sys.exit(app.exec_())
