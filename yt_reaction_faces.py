#!/usr/bin/env python
import os
import re
import cv2
import sys
import ast
import glob
import json
import time
import http.client as httplib
from shutil import copyfile
import getopt
import shutil
import tempfile
import requests
import operator
import numpy as np
import pandas as pd
from pytube import YouTube
from collections import defaultdict
from dplython import (DplyFrame, X, select, arrange, mutate, group_by, ungroup, summarize, sift)

ms_key1 = 'USE YOUR OWN KEY'

emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]

def downloadYtMp4(yURL, dlDir=os.getcwd()):
    #find youtube video
    yt = YouTube(yURL)
    #filter to only mp4 files and take last in list (sorted lowest to highest quality)
    hqMp4 = yt.filter('mp4')[-1]
    #strip quality from video info.. example video info:
    m = re.search("- (\d\d\dp) -", str(hqMp4))
    #save quality capturing group
    quality = m.group(1)
    #get mp4 video with highest quality found
    video = yt.get('mp4', quality)
    #download and save video to specified dir
    video.download(dlDir)

def mp4Frames(yURL, picDir, maxCount):
    dirpath = tempfile.mkdtemp()
    print("downloading yt video to tmp dir: %s" % dirpath)
    downloadYtMp4(yURL, dirpath)
    mp4File = glob.glob("%s/*.mp4" % dirpath)[0]
    print("Downloaded file: "+mp4File)
    #copying the file from temp to current working directory
    #this is to avoid file permission denied error
    copyfile(mp4File,"lol.mp4")
    mp4File = glob.glob("lol.mp4")[0]
    vidcap = cv2.VideoCapture(mp4File)
    success = True
    count = 0
    framesRead = 0
    while success:
        success, image = vidcap.read()
        if framesRead == maxCount:
            success = False
        if count%30 == 0:
            #trying to keep a tab on the frames read
            framesRead += 1
            #print 'Read a new frame: ', success
            if success:
                cv2.imwrite("%s/frame%d.jpg" % (picDir, count), image)     # save frame as JPEG file
        count += 1
    #when done - delete the temp file
    #the vudeo downloaded is saved in the current working directory
    shutil.rmtree(dirpath)

def getNewInstances(yURL, faceDet, faceDet2, faceDet3, faceDet4, maxCount):
    framepath = tempfile.mkdtemp()
    mp4Frames(yURL, framepath, maxCount)
    files = glob.glob("%s/*" %framepath) #Get list of all images with emotion
    prediction_data   = []
    predictDataSrcImg = []
    predictFaceDims   = []
    fileInd = 0
    for f in files:
        fileInd += 1
        print("detecting faces in frame: %d of %d" %(fileInd, files.__len__()))
        if f[-9:] != "Thumbs.db": #f windows
            #
            frame = cv2.imread(f) #Open image as grayscale
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
            #
            #Detect face using 4 different classifiers
            #this can be done by the MS Emotion api also, but you would want to keep the costs down
            #hence you use Haar Cascades
            #There are four of these files in the haarcascasde folder
            # other files can be found here: https://github.com/opencv/opencv/tree/master/data/haarcascades
            # opencv official link for haar cascading: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

            face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(face) == 1:
                facefeatures = face
            elif len(face2) == 1:
                facefeatures = face2
            elif len(face3) == 1:
                facefeatures = face3
            elif len(face4) == 1:
                facefeatures = face4
            else:
                facefeatures = None
            if facefeatures is not None:
                #Cut and save face
                for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing the face
                    gray = gray[y:y+h, x:x+w] #Cut the frame to size
                    try:
                        out = cv2.resize(gray, (350, 350))
                        #Resize face so all images have same size
                        prediction_data.append(out)
                        predictDataSrcImg.append(frame)
                        predictFaceDims.append([x, y, w, h])
                    except:
                       pass #If error, pass file
    #once done, remove all temporary files
    shutil.rmtree(framepath)
    return prediction_data, predictDataSrcImg, predictFaceDims

def processRequest(image, headers):
    tempDir = tempfile.mkdtemp()
    ms_api = 'https:://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'
    #ms api link
    imagePath = "%s/singleFrame.jpg" % tempDir
    cv2.imwrite(imagePath, image)
    with open(imagePath, 'rb') as f:
        data = f.read()
    shutil.rmtree(tempDir)
    json = None
    params = None
    retries = 0
    result = None
    while True:
        try:
            conn = httplib.HTTPSConnection('westus.api.cognitive.microsoft.com')
            conn.request("POST", "/emotion/v1.0/recognize", data, headers)
            response = conn.getresponse()
            response_data = response.read()
            conn.close()
            response_data=str(response_data)
            #get the response data in binary form
            response_data=response_data[2:(len(response_data)-1)]
            #convert to string and then extract only the required data
            result = ast.literal_eval(response_data)
            #convert string representation to actual datatype - list of dictionaries
            break
        except Exception as e:
            print("[Errno {0}]".format(e))
            if retries <= 5:
                time.sleep(1)
                retries += 1
                continue
            else:
                print( 'Error: failed after retrying!' )
                result = None
                break
    # print(type(result))
    return result

def main(argv):
    yURL = None
    outdir = None
    maxFrames = 500
    yURL=input("Enter the youtube url:")
    outdir = input("Enter the output directory:")
    maxFrames=int(input("Enter the maximum number of frames to check:"))

    faceDet  = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
    faceDet2 = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_alt_tree.xml")
    #
    pdata, pframes, pfacedims = getNewInstances(yURL, faceDet, faceDet2, faceDet3, faceDet4, maxCount=maxFrames)
    #
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = ms_key1
    headers['Content-Type'] = 'application/octet-stream'
    #
    resultsDf = pd.DataFrame()
    frameId = 0
    for image in pframes:
        print ("posting frame %d of %d" %(frameId,len(pframes)))
        #sending the frame image to MS cognitive services
        resultMS = processRequest(image, headers)
        #isinstance == type()
        if isinstance(resultMS, list):
            for result in resultMS:
                if isinstance(result, dict):
                    resFrameList = []
                    for res in result['scores'].items():
                        resFrameList.append((frameId,res[0],res[1],
                                             result["faceRectangle"]['left'],
                                             result["faceRectangle"]['top'],
                                             result["faceRectangle"]['width'],
                                             result["faceRectangle"]['height']))
                        appendDf = pd.DataFrame(resFrameList, columns=["frameId", "emotionLabel", "conf", "faceleft","facetop","faceW","faceH"])
                        resultsDf = resultsDf.append(appendDf)
        time.sleep(2)
        frameId += 1
    #
    # print(resultsDf)
    #we append all the data to the dataframe
    #http://bluescreen.club/2017/06/18/import-pandas-as-pd/
    #then we convert the dataframe to a Dplyframe object which allows us to do higher level data analytics
    #for this one, we will select out the top most ranking face frames for each of the emotions
    #microsoft provides us with around 8 emotions
    #so we sort out 8 faces for 8 emotions and then save them accordingly
    dfFaces = DplyFrame(resultsDf)
    # print(dfFaces)
    topFaces = (dfFaces >>
                   group_by(X.emotionLabel) >>
                   sift(X.conf == X.conf.max()) >>
                   sift(X.frameId == X.frameId.min()) >>
                   ungroup() >>
                   group_by(X.frameId) >>
                   sift(X.conf == X.conf.max()) >>
                   ungroup() >>
                   arrange(X.emotionLabel))

    topFaces = topFaces.drop_duplicates()
    #print(topFaces)
    i = 0
    for index, row in topFaces.iterrows():
        print ("saving emotion frame %d of %d" %(i,len(topFaces.index)))
        #
        emotion = row["emotionLabel"]
        confid  = int(row["conf"]*100)
        image   = pframes[int(row["frameId"])]
        faceL = row["faceleft"]
        faceT = row["facetop"]
        faceW = row["faceW"]
        faceH = row["faceH"]
        #save cropped face
        imageW = image[faceT:faceT+faceH, faceL:faceL+faceW]
        cv2.imwrite(os.path.expanduser("%s/Cropped_%s.jpg" % (outdir, emotion)), imageW)
        #if you wish to put a rectangle on the faces then uncomment below
        #
        # cv2.rectangle( image,(faceL,faceT),
        #               (faceL+faceW, faceT + faceH),
        #                color = (255,0,0), thickness = 5 )
        # cv2.putText( image, emotion, (faceL,faceT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1 )
        #
        cv2.imwrite(os.path.expanduser("%s/%s.jpg" % (outdir, emotion)), image)
        i += 1

if __name__ == "__main__":
   main(sys.argv[1:])
