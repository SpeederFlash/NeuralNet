import numpy as np
from random import shuffle
inputSet = []
outputSet = []

np.set_printoptions(threshold=np.nan)
for i in range(100):
    for j in range(100):
        for mads in range(2):
            #
            # Sorry no divides lol (or subtracts)
            #

            tempStringMADS = str(bin(mads))[2:]
            tempStringI = str(bin(i+1))[2:]
            tempStringJ = str(bin(j+1))[2:]
            listMADS = []
            listI = []
            listJ = []

            # MADS
            if(len(tempStringMADS)<1):
                for x in range(1-len(tempStringMADS)):
                    listMADS.append(0)

            for a in range(1):
                try:
                    listMADS.append(int(tempStringMADS[a]))
                except:
                    continue

            # I
            if(len(tempStringMADS)<8):
                for y in range(8-len(tempStringI)):
                    listI.append(0)

            for b in range(8):
                try:
                    listI.append(int(tempStringI[b]))
                except:
                    continue

            # J
            if(len(tempStringMADS)<8):
                for z in range(8-len(tempStringJ)):
                    listJ.append(0)

            for c in range(8):
                try:
                    listJ.append(int(tempStringJ[c]))
                except:
                    continue

            input = [listMADS[0], listI[0], listI[1], listI[2], listI[3], listI[4], listI[5], listI[6], listI[7],listJ[0], listJ[1], listJ[2], listJ[3], listJ[4], listJ[5], listJ[6], listJ[7]]
            inputSet.append(input)

            # FYI, largest number (10000) ends up as 17 binary digits
            if mads == 0:
                tempStr = str(bin(int(i+1)*int(j+1)))[2:]
                tempStrList = []
                if(len(tempStr)<18):
                    for incre in range(18-len(tempStr)):
                        tempStrList.append(0)
                for q in range(18):
                    try:
                        tempStrList.append(int(tempStr[q]))
                    except:
                        continue

                outputSet.append(tempStrList)

            if mads == 1:
                tempStr = str(bin(int(i+1)+int(j+1)))[2:]
                tempStrList = []
                if(len(tempStr)<18):
                    for incre in range(18-len(tempStr)):
                        tempStrList.append(0)
                for q in range(18):
                    try:
                        tempStrList.append(int(tempStr[q]))
                    except:
                        continue

                outputSet.append(tempStrList)

                outputSet.append(tempStrList)

temp = list(zip(inputSet, outputSet))

shuffle(temp)

inputSet, outputSet = zip(*temp)

inputSet = np.array(inputSet)
outputSet = np.array(outputSet).T


#
# Neural Network Time (woohoo)
#

class neuralNet():
    def __init__(self, numberOfLayer1, numberOfLayer2, numberOfLayer3, outputWeightsSize, trainingInputs, correctOutputs):

        np.random.seed(12)

        self.inputlength = len(trainingInputs[0])

        self.synapticWeightsListLayer1 = []
        self.synapticWeightsListLayer2 = []
        self.synapticWeightsListLayer3 = []
        self.outputWeightsList = []
        self.Layer1 = np.zeros(50,)
        self.Layer2 = np.zeros(50,)
        self.Layer3 = np.zeros(50,)
        self.LayerOutput = np.zeros(18,)

        for i in range(numberOfLayer1):

            # Gives one hidden layer node its weights

            self.synapticWeightsListLayer1.append((2 * np.random.random((self.inputlength,1)) - 1))

        for j in range(numberOfLayer2):

            self.synapticWeightsListLayer2.append((2 * np.random.random((numberOfLayer1,1)) - 1))

        for x in range(numberOfLayer3):

            self.synapticWeightsListLayer3.append((2 * np.random.random((numberOfLayer2,1)) - 1))

        for y in range(outputWeightsSize):

            self.outputWeightsList.append((2 * np.random.random((numberOfLayer3,1)) - 1))

        self.weightsOverseerList = [self.synapticWeightsListLayer1, self.synapticWeightsListLayer2, self.synapticWeightsListLayer3, self.outputWeightsList]
        self.layerOverseerList = [self.Layer1, self.Layer2, self.Layer3, self.LayerOutput]
        self.layerSizeOverseerList = [numberOfLayer1, numberOfLayer2, numberOfLayer3, outputWeightsSize]


    def __sigmoid(self, x):
        return (1/(1 + np.exp(-x)))

    def __sigmoidDeriv(self, x):
        return (x * (1-x))

    def train(self, inputs, trainingOutputs, iterations):
        for iteration in range(iterations):
            basis = [0.] * len(trainingOutputs[0])
            outputDeltas = np.array(basis)
            predictedOutputs = self.think(inputs)
            errorList = []
            for eleme in range(len(trainingOutputs)):
                errorSingle = []
                errorSingleList = []
                for i in range(len(trainingOutputs[eleme])):
                    errorSingle = trainingOutputs[eleme][i] - predictedOutputs[eleme][i]
                    errorSingleList.append(errorSingle)
                errorList.append(errorSingleList)

            ErrorList = np.array(errorList)

            for index in range(len(trainingOutputs)):

                output = trainingOutputs[index]
                sigDerivOfPredOut = self.__sigmoidDeriv(predictedOutputs[index])
                input = inputs[index]
                Delta = ErrorList[index]
                for j in range(len(Delta)):
                    add = Delta[j]
                    current = outputDeltas[j]
                    total = current + add
                    outputDeltas[j] = total

            for Layer in range(3):
                if Layer == 0:
                    nodeDeltas = [0.] * len(self.outputWeightsList[0])
                elif Layer == 1:
                    nodeDeltas = [0.] * len(self.synapticWeightsListLayer3[0])
                elif Layer == 2:
                    nodeDeltas = [0.] * len(self.synapticWeightsListLayer2[0])

                nodeDeltas = np.array(nodeDeltas)

                if Layer == 0:
                    for j in range(len(outputDeltas)):
                        delta = outputDeltas[j]
                        for a in range(len(nodeDeltas)):
                            nodeDeltas[a] += delta * self.outputWeightsList[j][a]
                elif Layer == 1:
                    for j in range(len(self.Layer3Delts)):
                        delta = self.Layer3Delts[j]
                        for a in range(len(nodeDeltas)):
                            nodeDeltas[a] += delta * self.synapticWeightsListLayer3[j][a]
                elif Layer == 2:
                    for j in range(len(self.Layer2Delts)):
                        delta = self.Layer2Delts[j]
                        for a in range(len(nodeDeltas)):
                            nodeDeltas[a] += delta * self.synapticWeightsListLayer2[j][a]


                if Layer == 0:
                    self.Layer3Delts = nodeDeltas
                elif Layer == 1:
                    self.Layer2Delts = nodeDeltas
                elif Layer == 2:
                    self.Layer1Delts = nodeDeltas

            print(self.Layer1Delts)

            # TODO: Use error list to create adjustments for the weights from deltas

        #-------------------------------------------
        #
        # NeuronThinking below
        #
        #-------------------------------------------

    def think(self, inputs):

        #-----------------------------
        # Returns predicted outputs
        #-----------------------------

        thinkOutputList = []
        tempList = []
        tempList2 = []
        for i in range(len(inputs)):
            elem = inputs[i]
            thinkOutput = self.__layerIter(elem)[3]
            tempList.extend(thinkOutput)
            first = 0 + 18 * i
            last = 17 + 18 * i
            tempList2 = tempList[first:last]
            thinkOutputList.append(tempList2)
        return np.array(thinkOutputList)

    def __layerIter(self, input):
        storeInput = input
        for elem in range(len(self.layerOverseerList)):
            self.layerOverseerList[elem] = self.__neuronIter(input, self.weightsOverseerList[elem], self.layerSizeOverseerList[elem], self.layerOverseerList[elem])
            input = self.layerOverseerList[elem]
        return self.layerOverseerList

    def __neuronIter(self, input, layerWeightsList, layersize, layer):
        for iter in range(layersize):
            layer[iter] = (self.__neuronThink(input, layerWeightsList[iter]))
        return layer

    def __neuronThink(self, input, weights):

        # one neuron

        dotResult = np.dot(input, weights)
        return self.__sigmoid(dotResult)

        #-------------------------------------------
        #
        # Adjustments below
        #
        #-------------------------------------------


# Initialize Network

# TODO: create layers, apply weights and adjustments, get outputs. check
#
# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
#
# OR
#
# https://www.ibu.edu.ba/assets/userfiles/it/2012/eee-ANN-5.pdf
#
# for help

neuralNetwork = neuralNet(50,50,50,18,[[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],[]).train(np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]), 1)
