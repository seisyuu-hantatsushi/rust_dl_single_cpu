use std::{ops};
use num;

use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

    pub fn neg(&self) -> Tensor<T> {
	let v = self.buffer().iter().map(|v| {num::zero::<T>() - *v}).collect::<Vec<T>>();
	Tensor::<T>::from_vector(self.shape().to_vec(), v)
    }
}
