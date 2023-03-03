use std::ops;
use num;

use crate::tensor::tensor_base::{Tensor,SubTensor};

pub struct TensorIter<'a, T>
where T:Clone {
    rt: &'a Tensor<T>,
    index: usize
}

impl<T> Tensor<T>
where T:Clone {
    pub fn iter<'a>(&'a self) -> TensorIter<'a, T> {
	TensorIter {
	    rt: self,
	    index: 0
	}
    }
}

impl<'a,T> Iterator for TensorIter<'a, T>
where T:Clone {
    type Item = SubTensor<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
	if self.index < self.rt.shape()[0] {
	    let current = self.index;
	    self.index += 1;
	    Some(self.rt.subtensor(current))
	}
	else {
	    None
	}
    }
}
