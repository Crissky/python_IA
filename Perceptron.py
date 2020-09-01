class Perceptron:
    def __init__(self, dataset, labels, threshold=0.2, learning_rate=0.01, epoches=100):
        self.dataset = dataset
        self.labels = labels
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.weights = [0]*len(dataset[0])

    def __str__(self):
        space = '\n';
        text = 'Informações do Perceptron' + space;
        text += "Limiar: " + str(self.threshold) + space;
        text += "Taxa de aprendizado: " + str(self.learning_rate) + space;
        text += "Entradas: " + str(self.dataset) + space;
        text += "Pesos: " + str(self.weights) + space;

        return text;
    def predict(self, inputs):
        product_vector = [x * y for x, y in zip(inputs, self.weights)]
        sumation = sum(product_vector)

        return self.toBinary(sumation)

    def toBinary(self, value):
        if(value >= self.threshold):
            activation = 1
        else:
            activation = 0

        return activation

    def fit(self):
        print("Iniciando Treinamento com {} épocas...".format(self.epoches))
        for epoch in range(self.epoches):
            for inputs, label in zip(self.dataset, labels):
                output = self.predict(inputs)
                delta_weight = list(map(lambda xi: self.learning_rate * (label - output) * xi, inputs))
                self.weights = [x + y for x, y in zip(self.weights, delta_weight)]
        print("Treinamento Concluído")
        

if(__name__ == '__main__'):
    training_inputs = [(0,0), (0,1), (1,0), (1,1)];
    labels = [0, 0, 0, 1]; #AND
    #labels = [0, 1, 1, 0]; #XOR
    threshold = 0.2;
    learning_rate = 0.01;
    
    perceptron = Perceptron(training_inputs, labels, threshold, learning_rate);
    print(perceptron)
    perceptron.fit()
    print("predict [0,0]:", perceptron.predict([0,0]));
    print("predict [0,0]:", perceptron.predict([0,1]));
    print("predict [0,0]:", perceptron.predict([1,0]));
    print("predict [1,1]:", perceptron.predict([1,1]));
    print();
    print(perceptron);
