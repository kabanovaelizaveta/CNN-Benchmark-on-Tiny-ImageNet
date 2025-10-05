# CNN Benchmark on Tiny ImageNet

## Objective
Train convolutional neural networks (CNNs) for multiclass image classification on **Tiny ImageNet**. Compare at least three architectures (**VGG16, ResNet50, EfficientNetB0**) using **transfer learning** and **fine-tuning**. Evaluate performance with metrics such as **Accuracy, F1 Score, AUC**, as well as efficiency indicators (parameters, memory, latency).  

## Dataset
- **Tiny ImageNet**  
  - 200 classes  
  - 100,000 training images, 10,000 validation images  
  - Original size: 64×64 pixels  
- Preprocessing: resized to 224×224 pixels
  
## Data Augmentation
Applied only to the training set:  
- Random horizontal flips  
- Random rotations (±10°)  
- Random zooming  
- Random translations  
- Conversion to tensor + one-hot labels  

Validation set: only resizing and one-hot encoding.  

## Models & Architecture
Pretrained models used: **VGG16, ResNet50, EfficientNetB0**  
- Preprocessing layer (`tf.keras.applications.preprocess_input`)  
- Dropout (0.3)  
- L2 regularization (0.0001)  
- Dense layers: ReLU (hidden) + Softmax (output)  

## Training Setup
- Optimizer: **Adam**, lr=0.0001  
- Loss: **Categorical Crossentropy**  
- Epochs: 15 (transfer learning) + 5 (fine-tuning)  
- Batch size: 32  
- Callbacks: **EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**  

## Evaluation Metrics
- **During training:** Loss, Top-1 Accuracy, Top-5 Accuracy, Training Time, Parameters, Memory, CPU/GPU Latency, Efficiency Ratio  
- **During testing:** Accuracy, F1 Score, AUC  

## Training Results (GPU NVIDIA A100)

| Model               | Mode | Loss | Top-1 Acc | Top-5 Acc | Params (M) | Memory (MB) | Latency CPU (ms) | Latency GPU (ms) |
|---------------------|------|------|-----------|-----------|------------|-------------|-----------------|-----------------|
| VGG16               | TL   | 1.72 | 0.57      | 0.81      | 14.8       | 57.4        | 121             | 28              |
| ResNet50            | TL   | 1.21 | 0.69      | 0.89      | 24.1       | 92.8        | 339             | 224             |
| EfficientNetB0      | TL   | 1.21 | 0.71      | 0.89      | 4.4        | 17.5        | 353             | 271             |
| VGG16 (Fine-tuned)  | FT   | 1.47 | 0.63      | 0.85      | 14.8       | 57.4        | 126             | 28              |
| ResNet50 (Fine-tuned)| FT  | 1.01 | 0.75      | 0.92      | 24.1       | 92.8        | 344             | 227             |
| EfficientNetB0 (FT) | FT   | 1.03 | 0.75      | 0.91      | 4.4        | 17.5        | 355             | 275             |

## Testing Results

| Model              | Accuracy | F1   | AUC   |
|--------------------|----------|------|-------|
| VGG16              | 0.575   | 0.573| 0.983 |
| ResNet50           | 0.697   | 0.697| 0.993 |
| EfficientNetB0     | 0.584   | 0.592| 0.984 |
| VGG16 (Fine-tuned) | 0.635   | 0.633| 0.988 |
| ResNet50 (FT)      | 0.755   | 0.754| 0.995 |
| EfficientNetB0 (FT)| 0.628   | 0.633| 0.988 |

## Saliency Maps
Gradient-based saliency maps were generated to interpret model predictions. These visualizations highlight which pixels most strongly influenced classification decisions.  

## Examples
### Training Curves
Example: Loss and Top-1/Top-5 Accuracy over epochs for ResNet50.

<img width="492" height="991" alt="image" src="https://github.com/user-attachments/assets/cd76f476-95a5-4a34-b537-8d5b6cf9c21d" /> 

### Sample Predictions
Example: Model predictions on validation images using EfficientNetB0.

<img width="1004" height="839" alt="image" src="https://github.com/user-attachments/assets/ab3fa931-e242-4e72-9e0b-6d1e030b895a" />

### Saliency Maps
Example: Saliency map highlighting important pixels for prediction.

<img width="937" height="488" alt="image" src="https://github.com/user-attachments/assets/130ae0af-9ca9-4cb4-9c98-0d88e5d548ba" />

## Conclusion
- **ResNet50 (fine-tuned)** achieved the best performance across all metrics (Top-1 Accuracy, F1, AUC) but required the most memory and had higher latency.  
- **EfficientNetB0** offered a strong balance between accuracy and efficiency, with far fewer parameters and low memory usage.  
- **VGG16** was fastest on GPU and moderate in memory consumption but underperformed in accuracy.  
- **Fine-tuning** improved all models by ~3–6% across evaluation metrics.  
- **Saliency maps** provided interpretability, showing the most relevant regions of images used by CNNs for decision-making.  
