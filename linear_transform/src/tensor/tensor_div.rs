use std::{ops};
use num;

use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

    pub fn div_rank0(lhs:&Tensor<T>, rhs:&Tensor<T>) -> Tensor<T> {
	assert_eq!(lhs.shape(), &[1,1]);
	assert_eq!(rhs.shape(), &[1,1]);
	Tensor::<T>::from_array(&[1,1], &[lhs[vec![0,0]]/rhs[vec![0,0]]])
    }
}
