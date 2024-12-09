import math
import nn
#we are gonna import numpy
import numpy as np
import util
###########################################################################
class NaiveBayesDigitClassificationModel(object):


####Naive Bayes model for digit classification.

    def __init__(self):
########Initialize the model with default parameters and placeholders for probabilities.
        self.conditionalProb = None
        self.prior = None
        self.features = None
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = True # Look at this flag to decide whether to choose k automatically ** use this in your train method **
                                    # we are checking if it to choose k automatically
        self.legalLabels = range(10)

    def train(self, dataset):

        # Here we are gonna train the model using the dataset, optionally tuning the smoothing parameter k.
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in dataset.trainingData for f in datum.keys()]))

        kgrid = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.5, 1, 5]
        if not self.automaticTuning: # if k is fixed, only try one value
            kgrid = [self.k]
        self.trainAndTune(dataset, kgrid)

    def trainAndTune(self, dataset, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters. The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        trainingData = dataset.trainingData
        trainingLabels = dataset.trainingLabels
        validationData = dataset.validationData
        validationLabels = dataset.validationLabels

        bestAccuracyCount = -1  # best accuracy so far on validation set
        #Common training - get all counts from training data
        #We only do it once - save computation in tuning smoothing parameter
        commonPrior = util.Counter()  # Prior probability over labels(counting over labels)
        #commonConditionalProb = util.Counter()  #Conditional probability of feature feat being 1 indexed by (feat, label)
        #commonCounts = util.Counter()  #how many time I have seen feature 'feat' with label 'y' whether inactive or active
        #bestParams = (commonPrior, commonConditionalProb, kgrid[0])  # used for smoothing part  trying various Laplace factors kgrid

        commonFeatureCounts = {}  # counts of features per label

        # Initialize counts
        for label in self.legalLabels:
            commonPrior[label] = 0
            commonFeatureCounts[label] = util.Counter()

        # looping for populating counts
        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = int(trainingLabels[i])
            #"*** YOUR CODE HERE to complete populating commonPrior, commonCounts, and commonConditionalProb ***"
            #util.raiseNotDefined()


            # Update prior count
            commonPrior[label] += 1
            # update feature counts
            for feature, value in datum.items():
                commonFeatureCounts[label][(feature, value)] += 1

        totalData = len(trainingData)

        # lets try different k values
        for k in kgrid:  # smoothing parameter tuning loop
            prior = util.Counter()
            conditionalProb = util.Counter()

            # Compute prior probabilities
            for label in self.legalLabels:
                prior[label] = float(commonPrior[label]) / totalData

            # lets compute conditional probabilities with Laplace smoothing
            for label in self.legalLabels:
                labelFeatureCounts = commonFeatureCounts[label]
                totalLabelFeatures = commonPrior[label]
                for feature in self.features:
                    for value in [0, 1]:
                        count = labelFeatureCounts.get((feature, value), 0) + k
                        totalCount = totalLabelFeatures + k * 2  # two possible values: 0 and 1
                        conditionalProb[(feature, label, value)] = float(count) / totalCount

            # Temporarily storing the parameters
            self.prior = prior
            self.conditionalProb = conditionalProb

            # evaluating performance on validation
            predictions = self.classify(validationData)
            accuracyCount = sum(int(predictions[i] == validationLabels[i]) for i in range(len(validationLabels)))

            print("Performance on validation set for k=%f: (%.1f%%)" % (
                k, 100.0 * accuracyCount / len(validationLabels)))
            if accuracyCount > bestAccuracyCount:
                bestParams = (prior.copy(), conditionalProb.copy(), k)
                bestAccuracyCount = accuracyCount

        # below we are setting the best parameters
        self.prior, self.conditionalProb, self.k = bestParams
        print("Best Performance on validation set for k=%f: (%.1f%%)" % (
            self.k, 100.0 * bestAccuracyCount / len(validationLabels)))

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis
        for datum in testData:
            # Compute the log joint probabilities for each label
            logJoint = self.calculateLogJointProbabilities(datum)
            # Find the label with the highest log probability
            bestLabel = logJoint.argMax()
            guesses.append(bestLabel)
            self.posteriors.append(logJoint)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for label in self.legalLabels:
            logProb = math.log(self.prior[label]) if self.prior[label] > 0 else float('-inf')
            for feature in self.features:
                value = datum[feature]
                prob = self.conditionalProb.get((feature, label, value), 1e-10)
                if prob > 0:
                    logProb += math.log(prob)
                else:
                    logProb += float('-inf')
            logJoint[label] = logProb
        return logJoint

################################################################################3
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        y = self.run(x)
        return 1 if nn.as_scalar(y) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                actual = nn.as_scalar(y)
                if prediction != actual:
                    self.w.update(nn.Constant(x.data * actual), 1)
                    converged = False

########################################################################33
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.
    """
    def __init__(self):
        # Initialize your model parameters here. Here you setup the architecture of your NN, meaning how many
        # layers and corresponding weights, what is the batch_size, and learning_rate.
        self.W1 = nn.Parameter(1, 40)
        self.b1 = nn.Parameter(1, 40)
        self.W2 = nn.Parameter(40, 30)
        self.b2 = nn.Parameter(1, 30)
        self.W3 = nn.Parameter(30, 1)
        self.b3 = nn.Parameter(1, 1)
        self.learning_rate = 0.01
        #3 layers

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # First hidden layer
        Z1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        A1 = nn.ReLU(Z1)
        # Second hidden layer
        Z2 = nn.AddBias(nn.Linear(A1, self.W2), self.b2)
        A2 = nn.ReLU(Z2)
        # Output layer
        Z3 = nn.AddBias(nn.Linear(A2, self.W3), self.b3)
        return Z3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        """
            Trains the model.
        """
        loss = float('inf')
        while loss > 0.01:
            for x, y in dataset.iterate_once(50):
                # Compute loss and gradients
                loss_node = self.get_loss(x, y)
                gradients = nn.gradients(loss_node, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                # Update parameters
                self.W1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.W2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
                self.W3.update(gradients[4], -self.learning_rate)
                self.b3.update(gradients[5], -self.learning_rate)
            # Compute total loss on the dataset
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            print(f"Current loss: {loss:.5f}")

##########################################################################
class DigitClassificationModel(object):
    """
    A second model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to classify each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        

        self.W1 = nn.Parameter(784, 250)
        self.b1 = nn.Parameter(1, 250)
        self.W2 = nn.Parameter(250, 150)
        self.b2 = nn.Parameter(1, 150)
        self.W3 = nn.Parameter(150, 10)
        self.b3 = nn.Parameter(1, 10)
        self.learning_rate = 0.1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        Z1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        A1 = nn.ReLU(Z1)
        Z2 = nn.AddBias(nn.Linear(A1, self.W2), self.b2)
        A2 = nn.ReLU(Z2)
        Z3 = nn.AddBias(nn.Linear(A2, self.W3), self.b3)
        return Z3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        accuracy = 0.0
        while accuracy < 0.976:
            for x, y in dataset.iterate_once(60):
                loss_node = self.get_loss(x, y)
                gradients = nn.gradients(loss_node, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                for param, grad in zip([self.W1, self.b1, self.W2, self.b2, self.W3, self.b3], gradients):
                    param.update(grad, -self.learning_rate)
            accuracy = dataset.get_validation_accuracy()
            print(f"Validation accuracy: {accuracy:.4f}")

###################################################################################
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_size = 350  # Hidden size for the RNN representation

        # Input-to-hidden and hidden-to-hidden parameters
        self.W1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W_hidden1 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_hidden1 = nn.Parameter(1, self.hidden_size)
        self.W_hidden2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_hidden2 = nn.Parameter(1, self.hidden_size)

        # Hidden-to-output parameters
        self.W_output = nn.Parameter(self.hidden_size, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))

        # Training hyperparameters
        self.learning_rate = 0.2
        self.min_learning_rate = 0.01
        self.decay_rate = 0.9

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the initial (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # Initialize hidden state
        batch_size = xs[0].data.shape[0]
        h = nn.Constant(np.zeros((batch_size, self.hidden_size)))

        # Process each character in the sequence
        for x in xs:
            h1 = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x, self.W1), nn.Linear(h, self.W_hidden1)), self.b_hidden1))
            h = nn.ReLU(nn.AddBias(nn.Linear(h1, self.W_hidden2), self.b_hidden2))

        # Output layer
        logits = nn.AddBias(nn.Linear(h, self.W_output), self.b_output)
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = self.learning_rate
        max_epochs = 50
        for epoch in range(max_epochs):
            for xs, y in dataset.iterate_once(128):  #Larger batch size for faster training
                #compute loss and gradients
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [
                    self.W1, self.b1,
                    self.W_hidden1, self.b_hidden1,
                    self.W_hidden2, self.b_hidden2,
                    self.W_output, self.b_output
                ])

                # Gradient clipping to stabilize training
                clipped_gradients = [nn.Constant(np.clip(grad.data, -5, 5)) for grad in gradients]

                #update parameters
                for param, grad in zip([self.W1, self.b1, self.W_hidden1, self.b_hidden1,
                                        self.W_hidden2, self.b_hidden2, self.W_output, self.b_output], clipped_gradients):
                    param.update(grad, -learning_rate)

            #validation accuracy
            accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1}: Validation Accuracy = {accuracy:.4%}")

            # Early stopping
            if accuracy >= 0.85:
                print(f"Training complete! Validation accuracy: {accuracy:.4%}")
                break

            #decay learning rate
            learning_rate = max(self.min_learning_rate, learning_rate * self.decay_rate)

