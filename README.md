# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the dataset, drop unnecessary columns (like ID), handle missing values, and apply Label Encoding to categorical features and the target (Segmentation).

### STEP 2: 
Split the data into training and testing sets, then normalize features using StandardScaler.

### STEP 3: 
Convert the processed data into PyTorch tensors and create DataLoaders for batch processing.

### STEP 4: 
Build a feedforward neural network with fully connected layers and ReLU activation, ending with a multi-class output layer.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer with forward pass, loss calculation, backpropagation, and updates.

### STEP 6: 
Evaluate using accuracy, confusion matrix, and classification report, and test predictions on sample input.

## PROGRAM

### Name:Iswarya P

### Register Number:212223230082

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

model = PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

### Dataset Information

<img width="1338" height="261" alt="image" src="https://github.com/user-attachments/assets/7cb0980e-8057-4db4-8dd0-99aa2ef43aa2" />


### OUTPUT

## Confusion Matrix
<img width="328" height="192" alt="image" src="https://github.com/user-attachments/assets/62a19c97-5ea5-4136-9c2a-26e22e5444c6" />


<img width="763" height="594" alt="image" src="https://github.com/user-attachments/assets/7f7f913f-9694-4804-8d96-414b4a7b87ce" />

## Classification Report
<img width="564" height="266" alt="image" src="https://github.com/user-attachments/assets/452b69c8-e7cf-404b-90de-66db017673d9" />


### New Sample Data Prediction
<img width="461" height="112" alt="image" src="https://github.com/user-attachments/assets/749250e7-11a6-44f8-848d-d5c21008818d" />


## RESULT
Thus developing a neural network classification model for the given data set has been executed successfully.
