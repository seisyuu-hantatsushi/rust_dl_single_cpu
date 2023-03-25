/* -*- tab-width:4 -*- */
use num;

use std::convert::{From,Into};
use crate::tensor::tensor_base::Tensor;

impl From<Tensor<f64>> for Tensor<f32> {
    fn from(x: Tensor<f64>) -> Self {
	let v = x.buffer().iter().map(|e| *e as f32).collect::<Vec<f32>>();
	Tensor::<f32>::from_vector(x.shape().to_vec(), v)
    }
}

