# Project Write-Up

This project can be run using the command

`python <Project_dir>/main.py -i <Project_dir>/resources/Pedestrian_Detect_2_1_1.mp4  -m <Project_dir>/model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml| ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

Remember to activate MQTT server, web app and FFmpeg sever.

## Explaining Custom Layers

Custom layer are layer that are not supported directly by the urret device which performs the convertion to IR. This layers must be handled in order to successfully get the IR of a model from an external framework, like Tensorflow. 

Most of the processes to handle this, is to create the corresponding extention, so the custom layer could run on them accordingly.

## Comparing Model Performance

I weighted the zip model before convertion and it was aprox. 190 MB, and after the .bin file was 68 MB, it reduced itself less than a half of its previous weight.

The inference time of the model pre- and post-conversion varied not too much, before conversion I obteined about 70 and 120 ms per frame, and after 60ms average, however, it was reduced clearly after convetion. I used time.time() variation to calculate that time.

The average accuracy was reduced after conversion about 15% and 20%, I used the confusion matrix method to measure that.

The costs using the cloud are: Internet dependent, higher latency, and higher security risks.  The edge  gives more realiability, and faster response.

## Assess Model Use Cases

In voting process, to manage how many people are in the vote machine and send warning if there are more tham it should be.

To manage access to a lab, because it can count how many people got in.

## Assess Effects on End User Needs

- There are many assess to consider, especially because the model accuraccy. The camera should be most horizontally posible.

- The camera should have a resolution than a least 300x300 px. 

- Enough light to the human to distinguish between objects.

## Model Research

Model used  _ssd_mobilenet_v2_coco_2018_03_29_ 
- I downloaded the model using  this command`wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
- I extracted it `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

- I convert it to IR `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`

This model did not seemed to accurate but I used through artifact in order to keep it useful. I mean, just detect a person if he/she remains more than one second in the video. I draw to bounding box with less accuracy but I only count them under the conditions listed.

- More than one  second present continuosly to say he/she  get in.

- More than one second not present to say he/she left.
