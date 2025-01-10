
---

# Transfer Learning for Alpaca Classification

This project demonstrates the use of transfer learning with the MobileNet architecture to classify images of alpacas using a small, specialised dataset. The Alpaca Dataset (Small), sourced from Kaggle, provides diverse images of alpacas in varying poses, lighting conditions, and backgrounds. The aim of this project is to fine-tune MobileNet, a pre-trained convolutional neural network, to achieve high performance in a niche image classification task with limited data.


---

## Project Overview

This project explores:

1. The principles of transfer learning.


2. Modifications to MobileNet for task-specific fine-tuning.


3. Data preprocessing techniques for small datasets.


4. Training, evaluation, and deployment of a lightweight model for image classification.




---

## Dataset

The Alpaca Dataset (Small) contains high-quality images of alpacas. It includes a variety of poses and backgrounds to challenge the model’s ability to generalise. The dataset was curated to mimic real-world scenarios where data availability is limited.

Source: [Kaggle Alpaca Dataset (Small)](https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small/data)


---

## Project Structure

- `dataset/`: Contains the Alpaca Dataset.

- `notebooks/`: Contains the main Jupyter Notebook where the analysis, training, and results are documented.

- `models/`: Saved weights and configurations of the trained MobileNet model.

- `src/`: Includes the `test_utils.py` file, which provides helper functions or classes used in the notebook.

- `results/`: Outputs such as evaluation metrics and test results.

- `README.md`: Project documentation (this file).



---

## Workflow

The project follows these steps:

1. **Data Collection:** Import and review the Alpaca Dataset.


2. **Data Preprocessing:** Resize, normalise, and augment the dataset for training.


3. **Model Preparation:** Load the pre-trained MobileNet model and adjust its layers for classification.


4. **Training:** Fine-tune the model on the Alpaca Dataset.


5. **Evaluation:** Assess the model using accuracy and loss metrics.


6. **Deployment:** Test the model on unseen data to validate its generalisation capability.




---

## Requirements

To run the project, you need the following:

- Python 3.8 or later

- TensorFlow 2.x

- NumPy

- Matplotlib

- Pandas

- Jupyter Notebook


Install dependencies using:
```Bash
pip install -r requirements.txt
```

---

## Results

The fine-tuned MobileNet achieved high accuracy on the Alpaca Dataset, demonstrating the power of transfer learning in adapting pre-trained models for small, task-specific datasets. Evaluation metrics and sample predictions are available in the results/ folder.


---

## Key Findings

- Transfer learning enables efficient training on small datasets.

- Data augmentation helps prevent overfitting.

- MobileNet’s lightweight architecture is ideal for resource-constrained tasks.



---

## Future Work

- Explore additional datasets to test model generalisation further.

- Implement real-time alpaca detection using the trained model.

- Experiment with alternative architectures like ResNet or EfficientNet.



---

## Acknowledgements

- [Deeplearning.ai](https://www.deeplearning.ai/) for the MobileNet tutorial.

- [Kaggle](https://www.kaggle.com/) for providing the Alpaca Dataset.

---

## License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
