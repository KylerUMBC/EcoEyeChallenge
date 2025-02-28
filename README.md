# EcoEyeChallenge
ECO-Sort- A garbage classifier using computer vision

# Description-
The goal of the program is to accurately classify types of waste and the environemntally ideal way to dispose of them.

We utilized *Resnet50*, a pre-trained model and PyTorch for a transfer learning approach. We believed the practical choice was to make use of a model that had already been trained on millions of images and adapt it for our purposes. We opted for PyTorch as opposed to tensorflow because of ease of installation and use, and simpler documentation. 

Participating in an Eco-coding challenge, it is only natural that we keep our environmental impact in mind. An additional benefit of transfer learning is its minimal environmental impact compared to training from scratch, using fewer energy and computational resources. Through colab's compute credit, we trained our model with NVIDIA A100 GPUS. The Ampere architecture of the GPUs is known for being far more energy efficient than its predecessors. We also utilized Google Colab, a cloud computing environment, committed to running its data centers on renewable energy.

Images within the dataset are resized to fit ResNet requirements, converted to tensors, and normalized for our model. We started with an 80-10-10 split of our dataset, for training, validation, and testing respectively.

# Process and Challenges-
Our initial goal was to put together a rough implementation of the framework we deceided on in our initial meeting. In our first iteration, we faced issues with abnormally high accuracy relative to the number of epochs we used, the cause of which was unknown. 

Our first wave of changes included implementing early stoppage with 50 epochs, and incorporating regularization. We continued to wrestle with seemingly incorrect accuracy reports, suggesting that we had data leakage. We attempted to resolve this by implementing StratifiedShuffleSplit for better class distribution and to hopefully prevent overlaps.

After still facing leakage the next day, we determined the issue was with our indexing relative to the complete dataset for our training and validation datasets, leading to image overlaps. We dabbled with GUI with Tkinter and Pillow, and tested classification of user-uploaded images, with some degree of success. While our timeframe was not enough to develop a dedicated front end, we hope to build on the user experience for this in the future.

# Installing and Running
First, download *ecosort.py* and *garbage_classification_modelL2ES.pth* and place them in the SAME directory. Then, in command line, run python3 ecosort.py --dir /dir/name/here --output /output/name/here .After which, the program will iterate through the data set, classifying each image based on its environmentally optimal disposal method.
