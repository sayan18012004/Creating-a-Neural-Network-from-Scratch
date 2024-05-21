
# Creating a Neural Network from scratch

I have used the MNIST dataset from Kaggle to train a neural network that detects numbers between 0-9 solely using Numpy library.

Dataset Link: https://www.kaggle.com/competitions/digit-recognizer/data

 ## Underlying Math

My neural network will feature a simple two-layer architecture:

- **Input Layer (\( a[0] \))**: This layer will consist of 784 units, corresponding to the 784 pixels in each 28x28 input image.

- **Hidden Layer (\( a[1] \))**: The hidden layer will have 10 units activated by the Rectified Linear Unit (ReLU) function.

- **Output Layer (\( a[2] \))**: This layer will contain 10 units, each representing one of the ten digit classes. It will be activated by the softmax function.

### Forward Propagation

1. **Calculation of Z[1]**
   - Formula: Z[1] = W[1]X + b[1]
   - Shape: Z[1] ~ 10 x m

2. **Activation of A[1]**
   - Formula: A[1] = gReLU(Z[1])
   - Shape: A[1] ~ 10 x m

3. **Calculation of Z[2]**
   - Formula: Z[2] = W[2]A[1] + b[2]
   - Shape: Z[2] ~ 10 x m

4. **Activation of A[2]**
   - Formula: A[2] = gsoftmax(Z[2])
   - Shape: A[2] ~ 10 x m

### Backward Propagation

1. **Calculation of dZ[2]**
   - Formula: dZ[2] = A[2] - Y
   - Shape: dZ[2] ~ 10 x m

2. **Calculation of dW[2]**
   - Formula: dW[2] = 1/m * dZ[2]A[1]T
   - Shape: dW[2] ~ 10 x 10

3. **Calculation of dB[2]**
   - Formula: dB[2] = 1/m Σ dZ[2]
   - Shape: dB[2] ~ 10 x 1

4. **Calculation of dZ[1]**
   - Formula: dZ[1] = W[2]T*dZ[2] .* g[1]'(z[1])
   - Shape: dZ[1] ~ 10 x m

5. **Calculation of dW[1]**
   - Formula: dW[1] = 1/m * dZ[1]A[0]T
   - Shape: dW[1] ~ 10 x 10

6. **Calculation of dB[1]**
   - Formula: dB[1] = 1/m Σ dZ[1]

### Parameter Updates

- Update of W[2]: W[2] := W[2] - αdW[2]
- Update of b[2]: b[2] := b[2] - αdb[2]
- Update of W[1]: W[1] := W[1] - αdW[1]
- Update of b[1]: b[1] := b[1] - αdb[1]

### Forward Propagation

- A[0] = X: 784 x m
- Z[1] ~ A[1]: 10 x m
- W[1]: 10 x 784 (as W[1]A[0] ~ Z[1])
- B[1]: 10 x 1
- Z[2] ~ A[2]: 10 x m
- W[2]: 10 x 10 (as W[2]A[1] ~ Z[2])
- B[2]: 10 x 1

### Backward Propagation

- dZ[2]: 10 x m (A[2])
- dW[2]: 10 x 10
- dB[2]: 10 x 1
- dZ[1]: 10 x m (A[1])
- dW[1]: 10 x 10
- dB[1]: 10 x 1
