
pip install --upgrade pip

pip install torch==2.9.0+cpu torchvision==0.24.0+cpu torchaudio==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu

pip install ipycanvas

pip install numpy

## 1. Load the Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Prepare the images (convert to numbers and normalize)
# We turn the images into 'Tensors' (math grids) and adjust the brightness.
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])

# 2. Download the MNIST dataset
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=True)


## 2. Build the Model

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1. Flatten the 28x28 image into a single line of 784 pixels
        self.flatten = nn.Flatten()
        
        # 2. Layer 1: 784 pixels in -> 50 neurons out
        self.hidden = nn.Linear(784, 50)
        
        # 3. Layer 2: 50 neurons in -> 10 digits out (0-9)
        self.output = nn.Linear(50, 10)
        
        # 4. ReLU (The "Spark"): A gatekeeper that helps the AI 
        # learn patterns
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

model = SimpleNet()


## 3. Train the Model

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # learning rate defined lr

# Let the AI study the dataset for 2 rounds (Epochs)
for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()      # Reset the math
        outputs = model(images)    # Make a guess
        loss = criterion(outputs, labels) # How "wrong" was it?
        loss.backward()            # Backpropagation: Find the mistake
        optimizer.step()           # Fix the weights
    print(f"Epoch {epoch+1} complete.")


## 4. Inference

# 1. Pick one random image
# Think of an 'iterator' as a waiter bringing one plate at a time from the kitchen.
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 2. Get the prediction
# We feed the image into the model to get its raw guesses (logits).
output = model(images)

# We use Softmax to turn those raw guesses into a percentage (Confidence).
probabilities = torch.nn.functional.softmax(output, dim=1)

# 'torch.max' finds the highest score and tells us which digit (0-9) it is.
confidence, prediction = torch.max(probabilities, 1)

# 3. Show the result
# We use 'squeeze' to flatten the image data so it's ready for Matplotlib to draw.
plt.imshow(images[0].numpy().squeeze(), cmap='gray')
plt.title(f"AI Guess: {prediction.item()} ({confidence.item()*100:.1f}%)")
plt.axis('off') # Hide the X and Y axis numbers
plt.show()


## 5. GUI for Number Prediction

from ipycanvas import Canvas
import ipywidgets as widgets
from IPython.display import display

# 1. Create the Drawing Area
# sync_image_data=True is CRITICAL! It tells the browser to share the pixels with Python.
canvas = Canvas(width=280, height=280, sync_image_data=True)
canvas.fill_style = 'black'
canvas.fill_rect(0, 0, 280, 280)
canvas.stroke_style = 'white'
canvas.line_width = 15
canvas.line_cap = 'round' # This makes the 'ink' look smooth

# --- The "Hand" Logic (Tracking your mouse) ---
drawing = False
last_x, last_y = 0, 0

def handle_mouse_down(x, y):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = x, y

def handle_mouse_move(x, y):
    global drawing, last_x, last_y
    if not drawing: return
    # Draw a line from where the mouse WAS to where it IS now
    canvas.stroke_line(last_x, last_y, x, y)
    last_x, last_y = x, y

def handle_mouse_up(x, y):
    global drawing
    drawing = False

# Attach these instructions to our canvas
canvas.on_mouse_down(handle_mouse_down)
canvas.on_mouse_move(handle_mouse_move)
canvas.on_mouse_up(handle_mouse_up)

print("Canvas Ready! You can now draw in the box below.")
display(canvas)


## 6. Prediction Engine

import io, torch, numpy as np
from PIL import Image

# 1. The Output Area (where the text will appear)
output_log = widgets.Output()

def on_predict_clicked(b):
    with output_log:
        output_log.clear_output()
        try:
            # A. Grab the pixels you drew
            image_data = canvas.get_image_data()
            img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
            
            # B. Shrink to 28x28 (AI only speaks 'tiny' image!)
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # C. Normalize: Convert pixels to a range of -1 to 1
            img_array = (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
            img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

            # D. The Moment of Truth
            model.eval()
            with torch.no_grad():
                logits = model(img_tensor)
                prob = torch.nn.functional.softmax(logits, dim=1)
                conf, pred = torch.max(prob, 1)

            print(f"RESULT: I think you drew a {pred.item()}!")
            print(f"CONFIDENCE: {conf.item()*100:.2f}%")
            
        except Exception as e:
            print(f"Error: {e}. Be sure to draw something first!")

# 2. Creating the Buttons
predict_btn = widgets.Button(description="Predict", button_style='success')
clear_btn = widgets.Button(description="Clear", button_style='danger')

# 3. Connecting the Buttons to our Logic
predict_btn.on_click(on_predict_clicked)
clear_btn.on_click(lambda b: canvas.fill_rect(0, 0, 280, 280))

# 4. Display the controls
display(widgets.HBox([predict_btn, clear_btn]), output_log)


# Reflection

## 1. Testing digits
## After testing numerous times, I learned that you have to make the numbers take up almost the
## entire space before the model accurately predicts the number. At 2 epochs, even after writing
## the number very large, the model still cannot give the correct answer.

## Drew 1 - Result: 1, Confidenece 86.08%
## Drew 7 - Resutlt: 7, 66.52% - took several attempts. Sometimes though it was 5
## Drew 8 - Result: 3, 47.45% - even after 10 attempts, never guessed 8

## 2. Stress Testing
## At two epochs, the model cannot handle smaller numbers. The model fails the stress testing. 
## Its possible that the numbers the model is trained on take up the entire space, and thus any
## nubmer that's smaller than that is not recognized. 

## There are exceptions. The model has an easier time recognizing numbers like 2 and 5, however
## fail to recognize 3, 8, 6, 9, and 1. However, recognizing 2 and 5 could be a fluke as it almost
## always defaults to answering 5 or 2. 


## 3. Sabotage
## With no middle line, after 4 attempts, the model recognized 7 at 47.85% confidence
## with the middle line, after 12 attempts, the model didn't recognize 7. Most common answers were 
## 1 or 4. Confident scores for 1 or 4 were 40-70%.

## There are likely no 7s with the middle line in the training data, thus the model had no experience
# matching the middle line 7 with other 7s. It associated the middle line with 4. 


## 4. Varation in Performance - Changing hyperparameters
# Can't find where learning rate is defined
# Changed epochs from 2 to 5 -> the model might have gotten worse at predicting numbers. Accuracy
# was terrible.

# Changed learning rate (lr) from 0.01 to 0.5 -> I think the code broke, because the predictions 
# didn't change. 
# Changed learning rate (lr) from 0.01 to 0.05 -> broken again
# Changed learning rate back to 0.01 and epochs to 2 -> the code is working again.
# Change epochs from 2 to 3 -> slight improvement? The model managed to predict for the first time

