use std::{ops};
use num;

use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+num::pow::Pow<T, Output = T>+Clone+Copy {

    pub fn pow_rank0(&self, p:T) -> Tensor<T> {
	assert_eq!(self.shape(), &[1,1]);
	Tensor::<T>::from_array(&[1,1], &[self[vec![0,0]].pow(p)])
    }
}
