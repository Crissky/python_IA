class Perceptron:
    def __init__(self, training_inputs=[], weights=[], labels=[], threshold=1, learning_rate=0.1):
        self.training_inputs = training_inputs;
        self.weights = weights;
        self.labels = labels;
        self.threshold = threshold;
        self.learning_rate = learning_rate;

    def __str__(self):
        space = '\n\n';
        text = '';
        text += "Limiar: " + str(self.threshold) + space;
        text += "Taxa de aprendizado: " + str(self.learning_rate) + space;
        text += "Entradas: " + str(self.training_inputs) + space;
        text += "Pesos: " + str(self.weights);

        return text;

    def checkInputs(self):
        len_first_input = len(self.training_inputs[0]);
        is_input = all(len(my_input) == len_first_input for my_input in self.training_inputs);
        if(not is_input):
            raise Exception('Suas ENTRADAS não tem o mesmo tamanho.');

    def checkVector(self):
        len_input = len(self.training_inputs[0]);
        len_inputs = len(self.training_inputs);
        len_weights = len(self.weights);
        len_labels = len(self.labels);
        self.checkInputs();
        if(len_input != len_weights):
            raise Exception('O TAMANHO das ENTRADAS ({}) é diferente do TAMANHO dos PESOS ({})'.format(len_input, len_weights));
        if(len_inputs != len_labels):
            raise Exception('O total de ENTRADAS ({}) é diferente do total de SAÍDAS ESPERADAS ({})'.format(len_inputs, len_labels));
        
    def productSingleVectors(self, my_input):
        new_vector = [x * y for x, y in zip(my_input, self.weights)];
        summation = sum(new_vector);
        
        return summation;
    
    def toBinaryOutput(self, value):
        result = 0;
        if(value >= self.threshold):
            result = 1;

        return result;

    def updateWeights(self, my_input, output, expected_output):
        result = False;
        if(output != expected_output):
            result = True;
            for index in range(len(my_input)):
                delta_weight = self.learning_rate * (expected_output - output) * my_input[index];
                self.weights[index] += delta_weight;

        return result;
    
    def predict(self, my_input):
        if(len(my_input) != len(self.weights)):
            raise Exception('O TAMANHO das ENTRADAS ({}) é diferente do TAMANHO dos PESOS ({})'.format(len(my_input), len(self.weights)));
        
        product = self.productSingleVectors(my_input);
        activation = self.toBinaryOutput(product);

        return activation;
                
    def train(self, num_epochs=-1):
        self.checkVector();
        current_epoch = 0;
        while(num_epochs !=0):
            all_in = True;
            current_epoch += 1;
            num_epochs -= 1;
            print('Época: {}'.format(current_epoch));
            for index in range(len(self.training_inputs)):
                my_input = self.training_inputs[index];
                product = self.productSingleVectors(my_input);
                output = self.toBinaryOutput(product);
                is_updated = self.updateWeights(my_input, output, self.labels[index]);

                if(all_in and is_updated):
                    all_in = False;

            print('PESOS: {}'.format(self.weights));
            if(all_in):
                break
        print("TERMINOU!!!\n");
            

if(__name__ == '__main__'):
    training_inputs = [(0,0), (0,1),(1,0),(1,1)];
    weights = [0.2, -0.1];
    labels = [0, 0, 0, 1];
    threshold = 0.2;
    learning_rate = 0.1;
    
    perceptron = Perceptron(training_inputs, weights, labels, threshold, learning_rate);
    perceptron.train(num_epochs=-1);
    print("predict [0,0]:", perceptron.predict([0,0]));
    print("predict [1,1]:", perceptron.predict([1,1]));
    print();
    print(perceptron);
