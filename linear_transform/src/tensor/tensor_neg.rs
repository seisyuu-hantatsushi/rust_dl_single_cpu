use num;

use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

    pub fn neg(&self) -> Tensor<T> {
	let v = self.buffer().iter().map(|&v| {num::zero::<T>() - v}).collect::<Vec<T>>();
	Tensor::<T>::from_vector(self.shape().to_vec(), v)
    }

}

impl<T> Tensor<T>
where T:num::Num+num::Float+Clone+Copy {
    pub fn abs(&self) -> Tensor<T> {
	let v = self.buffer().iter().map(|&v| v.abs()).collect::<Vec<T>>();
	Tensor::<T>::from_vector(self.shape().to_vec(), v)
    }
}
