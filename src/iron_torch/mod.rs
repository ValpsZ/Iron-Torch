pub mod matrix;
pub mod activation;

pub struct Layer<'a> {
    weights: matrix::Matrix,
    biases: matrix::Matrix,
    data: Option<matrix::Matrix>,
    activation: activation::Activation<'a>
}

impl Layer<'_> {
    pub fn new<'a>(input_size: usize, output_size: usize, activation: activation::Activation<'a>) -> Layer {
        Layer {
            weights: matrix::Matrix::random(input_size, output_size),
            biases: matrix::Matrix::random(input_size, output_size),
            data: None,
            activation
        }
    }

    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        if input.len() != self.weights.cols {
            panic!("Input length ({}) does not match the number of columns in the weights matrix ({})", input.len(), self.weights.cols);
        }

        let mut output = matrix::Matrix::from(vec![input]).transpose();
        output = self.weights
            .multiply(&output)
            .add(&self.biases)
            .map(self.activation.function);
        self.data = Some(output.clone());

        output.data[0].to_owned()
    }

    pub fn backwards(&mut self, outputs: Vec<f64>, targets: Vec<f64>, learning_rate: f64) {
        if targets.len() != outputs.len() {
            panic!("Length mismatch: targets length ({}) does not match outputs length ({})", targets.len(), outputs.len());
        }

        let mut parsed = matrix::Matrix::from(vec![outputs]);        
        let errors = matrix::Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = parsed.map(self.activation.derivativ).dot_product(&errors).map(&|x| x * learning_rate);

        match self.data {
            Some(ref mut data) => self.weights = self.weights.add(&gradients.multiply(&data.transpose())),
            None => panic!("Data is missing. Cannot perform the operation."),
        }
        self.biases = self.biases.add(&gradients);
    }
}
