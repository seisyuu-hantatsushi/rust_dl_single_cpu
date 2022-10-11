use num;
use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {
    pub fn square(&self) -> Tensor<T> {
	let b:&[T] = self.buffer();
	let v = b.iter().map(|b| (*b) * (*b)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}
