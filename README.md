# Flipkart GRiD Challenge: Object Localization

This repository contains the code and documentation for our solution to the Flipkart GRiD Challenge. The challenge involved developing a model to localize objects within images using a predefined dataset provided by Flipkart.

## Overview

The Flipkart GRiD Challenge is an annual campus engagement program organized by Flipkart. The challenge tasks participants with leveraging their machine learning and AI capabilities to develop a model for localizing objects within images.

## Problem Statement

Given a dataset consisting of images and metadata containing bounding box coordinates, the objective is to develop a model that can accurately localize objects within these images. The performance of the model is evaluated based on the Mean Intersection over Union (IoU) metric, which measures the overlap between predicted and ground truth bounding boxes.

## Approach

Our approach to solving the object localization challenge involved the following steps:

1. **Data Analysis and Visualization:** We analyzed the provided dataset and visualized images with bounding boxes to gain insights into the task.

2. **Model Selection:** We experimented with various pre-trained models available in Keras, such as MobileNetV2, DenseNet121, ResNet50V2, etc., and fine-tuned them for the object localization task.

3. **Model Training:** We trained each model separately using data augmentation techniques to improve generalization.

4. **Ensemble Method:** We combined the predictions of multiple models using an ensemble approach to improve overall performance.

5. **Performance Evaluation:** We evaluated the performance of our ensemble model on a blind test dataset provided by Flipkart using the Mean IoU metric.

## Results

Our ensemble model achieved a competitive Mean IoU score on the test dataset, demonstrating its effectiveness in localizing objects within images. Detailed results and performance metrics can be found in the Jupyter Notebooks.

## Acknowledgments

We would like to express our gratitude to Flipkart for organizing the GRiD Object Localization Challenge and providing the dataset for this project.
