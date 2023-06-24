use rand::prelude::*;


fn calc(a: &Vec<f32>, b: &Vec<f32>, c: &Vec<f32>) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((&x, &y), &z)| x.mul_add(y, z).tanh())
        .collect()
}

pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl Layer {
    pub fn new(size: usize) -> Layer {
        let weights: Vec<f32> = (0..size)
            .map(|_| rand::thread_rng().gen::<f32>())
            .collect();

        let biases: Vec<f32> = (0..size)
            .map(|_| rand::thread_rng().gen::<f32>())
            .collect();

        return Layer { weights, biases };
    }

    pub fn calc(&self, input: Vec<f32>) -> Vec<f32>{
        let output: Vec<f32> = calc(&self.weights, &input, &self.biases);

        return output;
    }
}
