#[allow(unused)] 
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[allow(unused)] 
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

#[allow(unused)] 
pub fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

#[allow(unused)] 
pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

#[allow(unused)] 
fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

#[allow(unused)] 
fn elu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

#[allow(unused)] 
fn softmax(input: &[f64]) -> Vec<f64> {
    let max_input = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = input.iter().map(|&x| (x - max_input).exp()).collect();
    let sum_exp = exp_values.iter().sum::<f64>();
    exp_values.iter().map(|&x| x / sum_exp).collect()
}

#[allow(unused)] 
fn tanh(x: f64) -> f64 {
    x.tanh()
}
