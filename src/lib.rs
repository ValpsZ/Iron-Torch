use rand::Rng;

pub struct Neuron {
    weight: f32,
    bias: f32,
}

impl Neuron {
    pub fn new() -> Neuron {
        let weight: f32 = rand::thread_rng().gen();
        let bias: f32 = rand::thread_rng().gen();
        return Neuron { weight, bias }
    }

    pub fn calc(&self, input: &Vec<f32>) -> f32 {
        let mut sum: f32 = 0.0;
        for value in input {
            sum += value * self.weight
        }
        let output: f32 = f32::tanh(sum) + self.bias;
        return output;
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(size: usize) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(size);
        
        for _ in 0..size {
            neurons.push(Neuron::new());
        }

        return Layer { neurons };
    }

    pub fn calc(&self, input: Vec<f32>) -> Vec<f32>{
        let mut output: Vec<f32> = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            output.push(neuron.calc(&input))
        }
        return output;
    }
}