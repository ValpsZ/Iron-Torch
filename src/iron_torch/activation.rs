use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivativ: &'a dyn Fn(f64) -> f64,
}

pub const TANH: Activation = Activation {
    function: &|x| (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x)),
    derivativ: &|x| 1.0 - ((E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x))) * ((E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x))),
};
