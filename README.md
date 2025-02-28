# EcoEyeChallenge
ECO-Sort - A garbage classifier using computer vision

# Description-
The goal of this project is to accurately classify types of waste and determine the environmentally ideal way to dispose of them.

We utilized *ResNet50*, a pre-trained model, and PyTorch to implement a transfer learning approach. By leveraging a model already trained on millions of images, we minimized training time and computational costs while maintaining high accuracy. We chose PyTorch over TensorFlow due to its ease of installation, user-friendly syntax, and well-structured documentation. 

As part of an eco-conscious coding challenge, we prioritized minimizing our environmental impact. Transfer learning consumes significantly less energy compared to training a model from scratch. Additionally, we used Google Colab, a cloud computing platform committed to renewable energy. We trained our model using NVIDIA A100 GPUs, known for their energy-efficient Ampere architecture, further reducing our carbon footprint. 

To prepare our dataset, images were resized to fit ResNet input requirements, converted to tensors, and normalized. We used an 80-10-10 dataset split for training, validation, and testing, respectively.

# Process and Challenges-
Our initial goal was to build a rough prototype based on our agreed-upon framework. However, in our first iteration, we encountered abnormally high accuracy relative to the number of training epochs, indicating a potential issue with data leakage.

![image](https://github.com/user-attachments/assets/aff32f35-c723-4ad7-a6a9-2b7e33917d36)

To address this, we implemented early stopping at 50 epochs and incorporated regularization techniques. Despite these adjustments, the suspiciously high accuracy persisted, suggesting the same data leakage issue. We then applied StratifiedShuffleSplit to improve class distribution and prevent dataset overlaps.

![image](https://github.com/user-attachments/assets/1349a03e-6079-4452-ae76-692cf9a2210d)
![image](https://github.com/user-attachments/assets/0ef3876d-3f8e-4835-8b93-2f3939ffacaa)

After further debugging, we discovered that the issue stemmed from incorrect dataset indexing, causing unintended image overlaps between training and validation sets. Fixing this resolved the issue, and we achieved accurate predictions.

To enhance usability, we developed a GUI using Tkinter and Pillow, allowing users to upload images and receive real-time classification results. Testing demonstrated a high success rate in classifying user-uploaded images.

![image](https://github.com/user-attachments/assets/42a8b8cb-43b9-4504-b3ea-d9ae85578446)

# Installing and Running
First, open a terminal window and clone the repository by running:

git clone https://github.com/KylerUMBC/EcoEyeChallenge

then,

cd EcoEyeChallenge

To classify images in a directory and display predictions via the GUI, run:

python ecosort.py --dir /dir/path/here 

To process images without the GUI and save predictions to text and CSV files, run:

ecosort.py --dir /dir/name/here --output /output/dir/path/here

This will generate .txt and .csv files containing image names and their predicted classifications.

# Credits
Contributors - Kyler Gelissen, Isaiah Byrd, David Ameh
<br/>Coach - Jorge Martinez
