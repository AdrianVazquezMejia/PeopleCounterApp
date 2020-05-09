"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str,default = "/home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml",
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str, default = "/home/workspace/images/people-counter-image.png",
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boundingBox(result,frame, height,  width):
    arr = result.flatten()
    matrix =np.reshape(arr, (-1,7))
    for i in range(len(matrix)):
        if matrix[i][1]==1 and matrix[i][2]>0.1 :
            xmin = int(matrix[i][3]*width)
            ymin = int(matrix[i][4]*height)
            xmax = int(matrix[i][5]*width)
            ymax = int(matrix[i][6]*height)
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,0,255),1)
    return frame

def detect_person(result, people):
    global incident_flag, quantity, timesnap,timer, ticks
    arr = result.flatten()
    matrix =np.reshape(arr, (-1,7))
    persons = 0
    for i in range(len(matrix)):
        if matrix[i][1] ==1  and matrix[i][2]>0.1 :
            persons+=1
    if persons != quantity and persons > 0 and not incident_flag :
        timer = True
        ticks = 0
        quantity = persons
    if persons == 0 and incident_flag :
        timer = True
        quantity =0
    if persons != quantity and persons > 0 and incident_flag :
        ticks = 0
        timer= False
        quantity = persons
    if persons== 0 and not incident_flag:
        ticks = 0
        timer= False
        quantity = 0
    if timer :
        ticks +=1
    if ticks >15 :       
        timer = False
        ticks = 0
        if persons >0:
            incident_flag = True
            timesnap = timesnap /10;
            people = persons
            #print("People detected t {:.2f} s confidence {:.2f}".format(timesnap,matrix[0][2]))
        if persons == 0:
            incident_flag = False
            quantity = 0
            people = 0
            
    return people
    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    #prob_threshold = args.prob_threshold
    single_image = False
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model,args.device,CPU_EXTENSION)
    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()
    
    #Check for CAM, image or video
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image = True  
        input_stream = args.input
    else:
        input_stream = args.input
        if not os.path.isfile(args.input):
            log.error("Specified input file doesn't exist")
            sys.exit(1)
    
    cap = cv2.VideoCapture(input_stream)
    if input_stream :
        cap.open(args.input)
        
    if not cap.isOpened():
        log.error("Unable to open source")
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter('out.mp4', 0x00000021, 10, (width,height))
    #print("width is: {} and height is {}".format(width, height))
    global incident_flag, quantity, timesnap, timer, ticks
    incident_flag = False
    quantity = 0
    total = 0
    timesnap = 0
    timer = False
    ticks = 0
    curr_count = 0
    doneCounter = False
    start_time =0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        timesnap +=1
        flag, original_frame =cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###

        frame = cv2.resize(original_frame,(net_input_shape[3],net_input_shape[2]))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape(1, *frame.shape)
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_network(frame)
        ### TODO: Wait for the result ###
        inf_start = time.time()
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            out_frame = draw_boundingBox(result,original_frame, height,width)
            
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(out_frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            ### TODO: Calculate and send relevant information on ###
            out.write(out_frame)
            curr_count = detect_person(result, curr_count)
            if incident_flag and not doneCounter :
                start_time = time.time()
                total +=1
                doneCounter = True
                json.dumps({"total":total})
                #print("total is : {}".format(total))
                client.publish("person",json.dumps({"total":total}))
                #client.publish("person",json.dumps({"count":quantity}))
                
            if not incident_flag and doneCounter and total >= 1:
                doneCounter = False
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
            #print(curr_count)
            client.publish("person",json.dumps({"count":curr_count}))   
             #   incident_flag= False
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        ### TODO: Send the frame to the FFMPEG server ###
        if key_pressed == 27:
            break
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image:
            cv2.imwrite('out_image.jpg', out_frame)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
