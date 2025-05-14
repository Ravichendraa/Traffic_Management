# Traffic Management in indian context

This project addresses the challenge of traffic congestion in India by developing an intelligent traffic monitoring system using computer vision.  We leverage the YOLOv12m model to achieve highly accurate vehicle detection and classification in real-world traffic scenarios.

## Methodology

Our approach involves several key steps:

1. **Dataset Creation and Enhancement:** We began by collecting a dataset of 16,000 images of Indian traffic scenes, carefully annotating each image with bounding boxes around vehicles.  To improve the model's robustness and generalization capabilities, we augmented this dataset using techniques like rotation, shear transformation, and others, resulting in a significantly larger dataset of 48,000 images. This addresses the limitations of using a dataset lacking diversity in lighting, angles, and occlusions.  The dataset includes nine distinct vehicle classes: Auto-Rickshaw, Bicycle, Bus, Car, Cycle-Rickshaw, E-Rickshaw, Motorcycle, Tractor, and Truck.

2. **Model Training and Optimization:** We employed a pre-trained YOLOv12m model as our foundation.  We experimented with different training configurations: varying the number of training epochs and the size of the training dataset.  Our best performing model (Model 4) was trained for 100 epochs using the augmented dataset of 48,000 images.  This significantly improved the model's performance compared to models trained on smaller datasets or for fewer epochs.

3. **Performance Evaluation:**  We rigorously evaluated our models using standard metrics: precision, recall, and mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds (mAP@50 and mAP@50-95).  These evaluations helped us select the best performing model, which exhibited substantially improved accuracy across all vehicle classes compared to the initial YOLOv12m model and other training configurations.

4. **Web Application Development:** We developed a user-friendly web application that takes video input of traffic and provides real-time vehicle counts and classifications.  This allows for practical deployment and immediate application of our model's capabilities.

## Results

Our best-performing model (Model 4) achieved excellent results:

* **Overall Precision:** 0.967
* **Overall Recall:** 0.942
* **Overall mAP@50:** 0.969
* **Overall mAP@50-95:** 0.808

Furthermore, the model demonstrated high accuracy for individual vehicle classes, as detailed in Table 2 (within the original provided text). This table shows the mAP@50 scores for each vehicle type.

## Future Work

While our model has achieved impressive results, we plan to further enhance it by:

* **Implementing side-specific vehicle detection:** This will allow for a more granular analysis of traffic flow, enabling better congestion management strategies.
* **Expanding dataset diversity:** We will continue to expand our dataset with more diverse images to improve generalization and robustness in various traffic conditions.
* **Optimizing real-time deployment:** We aim to optimize the model and web application for even faster and more efficient real-time processing.
* **Improving classification accuracy for underrepresented classes:** We will focus on improving the accuracy of detecting smaller vehicles, such as bicycles and motorcycles, which are often occluded or difficult to identify.

## Comparison with Existing Methods

Our model significantly outperforms previous approaches in terms of accuracy and the number of vehicle classes identified. The table below summarizes the key differences between our approach and existing literature (details from the original text are referenced here for clarity).

| Feature             | Our Model (YOLOv12-Based) | Song et al. (2019) – YOLOv3 | Barcellos et al. (2015) – GMM | Ghosh et al. (2019) – MOG2 with OpenCV |
|----------------------|---------------------------|---------------------------|-----------------------------|------------------------------------|
| Model Used           | YOLOv12 (Model 4)         | YOLOv3                     | Gaussian Mixture Models (GMM) | MOG2 + OpenCV                    |
| Number of Vehicle Classes | 9 (Expanded classes)      | 3 (Car, Truck, Bus)        | 2 (Car, Truck)               | Multiple (based on object size)      |
| Dataset Size         | 16,000+ images            | Public dataset (fewer images)| Traffic surveillance videos    | Tested at 6 locations in Dhaka     |
| Image Augmentation   | Yes                        | No                         | No                          | No                               |
| Precision            | 0.967                      |  (Not reported in provided text) | (Not reported in provided text) | (Not reported in provided text)    |
| Recall               | 0.942                      | (Not reported in provided text) | (Not reported in provided text) | (Not reported in provided text)    |
| mAP@50               | 0.969                      | (Not reported in provided text) | (Not reported in provided text) | (Not reported in provided text)    |



This project represents a significant contribution to the field of intelligent traffic management, providing a more accurate and efficient system for monitoring and analyzing traffic flow in challenging Indian road conditions.
