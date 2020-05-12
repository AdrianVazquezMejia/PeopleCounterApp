#!/usr/bin/env python3
#python main.py -i /home/workspace/resources/Pedestrian_Detect_2_1_1.mp4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        
        self.exec_net = None
        self.plugin = None
        self.net = None
        self.input_blob = None
        self.output_blob = None
        return

    def load_model(self, model, device = "CPU", cpu_extension = None):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0]+".bin"
        self.plugin = IECore()
        ### TODO: Load the model ###
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension,device)
            
        self.net = IENetwork(model = model_xml, weights = model_bin)
        self.exec_net= self.plugin.load_network(self.net,device)
        ### TODO: Check for supported layers ###
        if device == "CPU":
            supported_layers = self.plugin.query_network(self.net, device)
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers ]
            if len(not_supported_layers) != 0:
                log.error("It seems that there are layer that are not supported by the device {}:\n {}".format(self.plugin.device, ',  '.join(not_supported_layers)))
                sys.exit(1)
        ### TODO: Add any necessary extensions ###

        ### TODO: Return the loaded inference plugin ###
        
        ### Note: You may need to update the function parameters. ###
        return self.plugin

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        self.input_blob = next(iter(self.net.inputs))
        shape = self.net.inputs[self.input_blob].shape
        return shape

    def exec_network(self, image):
        ### TODO: Start an asynchronous request ###
        self.exec_net.start_async(request_id = 0, inputs = {self.input_blob: image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_net.requests[0].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        self.output_blob = next(iter(self.net.outputs))
        output = self.exec_net.requests[0].outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###
        return output
