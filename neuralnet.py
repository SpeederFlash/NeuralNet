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

            # FYI, largest number (10000) ends up as 14 binary digits
            if mads == 0:
                tempStr = str(bin(int(i+1)*int(j+1)))[2:]
                tempStrList = []
                if(len(tempStr)<15):
                    for incre in range(15-len(tempStr)):
                        tempStrList.append(0)
                for q in range(15):
                    try:
                        tempStrList.append(int(tempStr[q]))
                    except:
                        continue

                outputSet.append(tempStrList)

            if mads == 1:
                tempStr = str(bin(int(i+1)+int(j+1)))[2:]
                tempStrList = []
                if(len(tempStr)<15):
                    for incre in range(15-len(tempStr)):
                        tempStrList.append(0)
                for q in range(15):
                    try:
                        tempStrList.append(int(tempStr[q]))
                    except:
                        continue

                outputSet.append(tempStrList)

                outputSet.append(tempStrList)

shuffle(inputSet)
inputSet = np.array(inputSet)
outputSet = np.array(outputSet).T


#
# Neural Network Time (woohoo)
#

class neuralNet():
    def __init__(self, numberOfLayer1, numberOfLayer2, numberOfLayer3, outputWeights):

        np.random.seed(12)

        self.synapticWeightsListLayer1 = []
        self.synapticWeightsListLayer2 = []
        self.synapticWeightsListLayer3 = []
        self.outputWeightsList = []

        for i in range(numberOfLayer1):

            # Gives one hidden layer node its weights

            self.synapticWeightsListLayer1.append((2 * np.random.random((18,1)) - 1))

        for j in range(numberOfLayer2):

            self.synapticWeightsListLayer2.append((2 * np.random.random((numberOfLayer1,1)) - 1))

        for x in range(numberOfLayer3):

            self.synapticWeightsListLayer3.append((2 * np.random.random((numberOfLayer2,1)) - 1))

        for y in range(outputWeights):

            self.outputWeightsList.append((2 * np.random.random((numberOfLayer3,1)) - 1))



    def __sigmoid(self, x):
        return (1/(1 - np.exp(x)))

    def __sigmoidDeriv(self, x):
        return (x * (1-x))

    def train(self, inputs, trainingOutputs, iterations):
        pass

    def think(self, inputs):
        pass

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
neuralNetwork = neuralNet(50,50,50,15)

print(inputSet)
