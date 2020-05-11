# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Custom layer are layer that are not supported directly by the urret device which performs the convertion to IR. This layers must be handled in order to successfully get the IR of a model from an external framework, like Tensorflow. 

Most of the processes to handle this, is to create the corresponding extention, so the custom layer could run on them accordingly.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

Model used  _ssd_mobilenet_v2_coco_2018_03_29_ 
- I downloaded the model using  this command`wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
- I extracted it `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

- I convert it to IR `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`

This model did not seemed to accurate but I used through artifact in order to keep it useful. I mean, just detect a person if he/she remains more than one second in the video. 

- 
