Flipkart GRiD - Te[a]ch the Machines AI/ML Challenge 2019
===========================================================

Please go through the jupyter notebook `Flipkart_Main.ipynb` for the code and detailed explanation.

Our configuration for training was Google Compute Cloud with 6 Cores Xeon, 32 GB RAM and Nvidia V100 GPU with 16 GB DDR5.

All other requirements such as packages and libraries are mentioned in the Notebook itself.


Since the task required just single object localization, we started off with basic imagenet architectures and finally made an ensemble of predictions to improve the overall score. We used an ensemble of MobileNetV2, InceptionResnetV2 and U-Net.

Imagenet Architectures
-----------------------
Since the task involved localization of just single object, the current imagenet architectures becomes a normal choice because of their accuracy and ease of implementation. So we decided to modify the top layers of these architectures to get regression output for the coordinates of bounding box.

We decided to use two different models, one with high depth and one with low depth to test what we obtain.

Since multiple competitions were won by both MobileNetV2 and InceptionResnetV2, they became our obvious choice.

Top Layer: We tried multiple combinations of top layer including Pooling, BatchNormalization and Fully-Connected Layers but found that the architectures worked best how they were designed and adding Convolutional Layers only helped. A possible reason we found for such outcome is also mentioned in this paper ( arXiv:1412.6806 ).

As expected both models performed well on the data with augmentations, with InceptionResnetV2 performing marginally better than MobileNetV2 because of the depth of the network.

We later added DenseNet121 to improve ensemble predictions as it is comparable in depth with MobileNetV2.

U-Net
------
Though this was an object localization challenge but segmentation has had a lot of research and lately pretty good models have emerged for the same.

We wanted to test segmentation out as a side-model, thus we chose U-Net as our model. U-Net is particularly small but pretty effective, which is exactly what we wanted. Other image segmentation models such DeepLabV3+ would have performed better but the overall training time would be huge.

We calculated the bounding box by determining the mask boundaries and then throwing out pixel values from the 640x480 matrix.

Since we couldn't have used other pre-built model for getting segmentation mask we used OpenCV and its advanced image operations to get a rough segmentation mask for the objects particularly effective where there was a good contrast in the background and the object. The method was ineffective against transparent/translucent objects and where there was a low contrast between the object and background.

Though we didn't achieve very high Mean IoU score the model gave pretty decent performance given the masks were roughly drawn and we noticed that it performed better than the imagenet models on high contrast images.

Thus we decided to include the model in our ensemble to get a better score overall.

Other Techniques
-----------------
Apart from the two already disccused we tried multiple architectures such as YOLOv2, RetinaNet and AttentionNet. We weren't really inclined towards region proposal networks as the task just included localization of the main single object.

These models plateau around ~0.86 Mean IoU score, and didn't help much.

AttentionNet did achieve a good score of ~0.90 combined with our MobileNetV2 as a base model, but we couldn't test it more due to time limitations.