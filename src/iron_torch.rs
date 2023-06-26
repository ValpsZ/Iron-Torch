mod matrix;
mod activation;

pub struct Layer<'a> {
    weights: matrix::Matrix,
    biases: matrix::Matrix,
    data: matrix::Matrix,
    activation: activation::Activation<'a>
}

impl Layer<'_> {
    fn new<'a>(size: usize, activation: activation::Activation<'a>) -> Layer {
        Layer {
            weights: matrix::Matrix::random(size, 1),
            biases: matrix::Matrix::random(size, 1),
            data: matrix::Matrix::zeros(size, 1),
            activation
        }
    }
    
    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        if input.len() != self.weights.cols {
            panic!("Input length ({}) does not match the number of columns in the weights matrix ({})", input.len(), self.weights.cols);
        }
        
        let mut output = matrix::Matrix::from(vec![input]).transpose();
        output = self.weights
            .multiply(&output)
            .add(&self.biases)
            .map(self.activation.function);
        self.data = output.clone();
        
        output.data[0].to_owned()
    }

    fn backwards(&mut self, outputs: Vec<f64>, targets: Vec<f64>, learning_rate: f64) {
        if targets.len() != outputs.len() {
            panic!("Length mismatch: targets length ({}) does not match outputs length ({})", targets.len(), outputs.len());
        }

        let mut parsed = matrix::Matrix::from(vec![outputs]);        
        let errors = matrix::Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = parsed.map(self.activation.derivativ).dot_product(&errors).map(&|x| x * learning_rate);

        self.weights = self.weights.add(&gradients.multiply(&self.data.transpose()));
        self.biases = self.biases.add(&gradients);
    }
}
