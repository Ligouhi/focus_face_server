from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from PIL import Image
from detector import detect_faces
import argparse
import requests as req
import re
from io import BytesIO
from PIL import ImageDraw
import cv2
import numpy as np

app = Flask(__name__)
api = Api(app)



parser = reqparse.RequestParser()
parser.add_argument('img')
parser.add_argument('xsplit',default=3)
parser.add_argument('ysplit',default=6)
parser.add_argument('nums',default=3)

class find_focus(Resource):
    def post(self):
        args = parser.parse_args()
        img_url = args['img']
        response = req.get(img_url)
        img = Image.open(BytesIO(response.content))
        bounding_boxes = detect_faces(img) # detect bboxes and landmarks for all faces in the image
        width, height = img.size
        #wnum:横向分割块数 hnum：纵向分割块数
        wnum = int(args['ysplit'])
        hnum = int(args['xsplit'])
        w_step = width/wnum
        h_step = height/hnum
        flag = np.ones((hnum,wnum))
        val = np.ones((hnum,wnum,2))
        res = []
        for i in range(0,len(bounding_boxes)):
            y = bounding_boxes[i][0]/(w_step+0.001)
            x = bounding_boxes[i][1]/(h_step+0.001)
            
            flag[int(x)][int(y)]+=1
            val[int(x)][int(y)][0]+=bounding_boxes[i][0]
            val[int(x)][int(y)][1]+=bounding_boxes[i][1]
        for i in range(0,int(args['nums'])):
            mpos = np.unravel_index(np.argmax(flag),flag.shape)
            res.append([val[mpos[0],mpos[1],1]/np.max(flag)/height,val[mpos[0],mpos[1],0]/np.max(flag)/width])
            flag[mpos[0],mpos[1]] = 0



        return res,201
       

    
api.add_resource(find_focus,'/focus')


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)