use std::{ops};
use num;

use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {
    pub fn scale(&self, s:T) -> Tensor<T> {
	let lhs:&[T] = self.buffer();
	let v = lhs.iter().map(|lhs| s * (*lhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

