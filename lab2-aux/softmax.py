import math
import torch
import numpy as np
import pickle
from lab2.activations import softmax

class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()
    
    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find torch.randn userful here *0.001
        self.W = torch.randn((self.input_shape, self.num_classes), requires_grad = True)
        #print(self.W.grad)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # 0. compute the dot product between the input X and the weight matrix
        # you can use @ for this operation
        # remember about the bias trick!
        # 1. apply the softmax function on the scores, see torch.nn.functional.softmax
        # think about on what dimension (dim parameter) you should apply this operation
        
        # 2. returned the normalized score

        # softmax auto-normalizes
        # dim 1 because we go along the columns (features)
        # so that softmax is applied independently to the scores of each class
        #print(X.matmul(self.W))
        scores = torch.nn.functional.softmax(X.matmul(self.W), dim = 1)
        return scores

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # TODO your code here
        # 0. compute the dot product between the input X and the weight matrix
        scores = X.matmul(self.W)
        #print(scores)
        # 1. compute the prediction by taking the argmax of the class scores
        # you might find torch.argmax useful here.
        # think about on what dimension (dim parameter) you should apply this operation
        
        # dim 1 because that's the features dimension
        # which gives the label of the max probability
        label = torch.argmax(scores, axis=1)
        return label

    def cross_entropy_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = -torch.log(y_pred.gather(1, y.unsqueeze(1)))
        loss = loss.mean()
        return loss

    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        # directly calculating log⁡(∑jexj)log(∑j​exj​) can be numerically unstable4
        # use the trick of subtracting the maximum value (max⁡jxjmaxj​xj​) before exponentiating and summing
        # "log-sum-exp" trick
        c = x.max()
        logsumexp = np.log(np.exp(x - c).sum())
        return x - c - logsumexp

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print('hyperparameters: lr {:.4f}, reg {:.4f}, epochs {:.2f}'.format(lr, reg_strength, epochs))

        for epoch in range(epochs):
            for ii in range((X_train.shape[0] - 1) // bs + 1):  # in batches of size bs
                # TODO your code here
                #print(self.W.grad)
                start_idx = ii * bs  # we are ii batches in, each of size bs
                end_idx = (ii + 1) * bs  # get bs examples
                # get the training training examples xb, and their coresponding annotations
                xb = X_train[start_idx:end_idx]
                yb = y_train[start_idx:end_idx]

                # apply the linear layer on the training examples from the current batch
                pred = None
                pred = self.predict_proba(xb)

                # compute the loss function
                # also add the L2 regularization loss (the sum of the squared weights)
                loss = torch.nn.functional.cross_entropy(pred, yb)
                history.append(loss.item())
                #optimizer.zero_grad()
                
                # start backpropagation: calculate the gradients with a backwards pass
                loss.backward()
                #print(self.W.grad)

                # update the parameters
                with torch.no_grad():  # we don't want to track gradients
                    # take a step in the negative direction of the gradient, the learning rate defines the step size
                    #sys.stdout.write('\rWeights: {}'.format(self.W))
                    self.W -= self.W.grad * lr

                    # ATTENTION: you need to explictly set the gradients to 0 (let pytorch know that you are done with them).
                    self.W.grad.zero_()

        return history

    def get_weights(self, img_shape) -> np.ndarray:
        # Check if the weight matrix is initialized and has the correct shape

        # Ignore the bias term
        weights = self.W[:-1, :].detach().numpy()

        # Reshape the weights to (*image_shape, num_classes)
        weights_reshaped = weights.reshape((*img_shape, self.num_classes))

        # Transpose the weights for proper visualization
        weights_transposed = weights_reshaped.transpose(3, 0, 1, 2)

        #print(weights_transposed)
        return weights_transposed

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        # you might find torch.load useful here
        with open(path, "rb") as f:
          data = pickle.load(f)

        self.W = data[2]
        # don't forget to set the input_shape and num_classes fields
        self.num_classes = data[1]
        self.input_shape = data[0]
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find torch useful for this
        # TODO your code here
        with open(path, "wb") as f:
          pickle.dump((self.input_shape, self.num_classes, self.W), f);
        return True

