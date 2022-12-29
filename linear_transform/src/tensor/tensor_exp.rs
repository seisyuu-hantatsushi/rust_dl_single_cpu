use num;
use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Float+Clone+Copy {
    pub fn exp(&self) -> Tensor<T> {
	let b:&[T] = self.buffer();
	let v = b.iter().map(|b| b.exp()).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

impl<T> Tensor<T>
where T:num::Float+Clone+Copy {
    pub fn sigmoid(&self) -> Tensor<T> {
	let one:T = num::one();
	let b:&[T] = self.buffer();
	let v = b.iter().map(|b| one/(one - b.exp())).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}
