
# From Convolution to Attention

This project focuses on training a Vision Transformer (ViT) on a specialized road dataset. The dataset is meticulously curated to include images with and without cars, making it a valuable resource for tasks related to road scene analysis. Vision Transformers have proven to be highly effective in various computer vision tasks, and this project harnesses their power to detect and understand the presence of cars on roads.

## Structure:

1. #### data.py 🧰
Responsible for data preprocessing.
Manages the road dataset, categorizing images into two classes: with cars 🚗 and without cars 🛑.

2. #### ViT.py 🤖
Implements the Vision Transformer (ViT) model architecture.
Leverages attention mechanisms for feature extraction, showcasing the shift from traditional convolutions to attention.

3. #### run.py 🏃‍♂️
The heart of the project that initiates model training.
Compares the performance of the ViT model against a baseline CNN model.
Evaluate and record key metrics to understand the efficacy of attention-based approaches.

4. #### CNN.py 🔄
Implements a Convolutional Neural Network (CNN) for benchmarking purposes.
It serves as a reference point to highlight the advantages of the Vision Transformer.

5. #### hyperparameters.json ⚙️
Contains a JSON file specifying hyperparameters for easy configuration and experimentation.

6. #### Data Folder 📁
Contains two folders named 'vehicles' and 'non-vehicles' which contain the dataset's images.

7. #### Requirement.txt 🛠️
Lists all the dependencies necessary for running the project. You can use this to set up your environment.



## Installation

1. #### Create a virtual python environment.

2. #### Clone the Repo

```bash
  git clone
```
3. #### Install required packages

```bash
  pip install -r requirements.txt
```

4. #### Train the Model

```bash
  python run.py
```
## Results

### Accuracy Graph for Vision Transformer

![App Screenshot](https://github.com/VibhuRaj01/From-Convolution-to-Attention/blob/main/Img/ViT%20accuracy.JPG)
🔵-Traning Data

🟠-Validation Data

### Accuracy Graph for CNN

![App Screenshot](https://github.com/VibhuRaj01/From-Convolution-to-Attention/blob/main/Img/CNN%20accuracy.JPG)
🔴-Training Data

🔵-Validation Data

## Support

For support, email me at vibhuraj2002@gmail.com.

