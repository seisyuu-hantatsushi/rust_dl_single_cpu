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
	let current = if self.rt.shape().len() <= 2 {
	    if self.rt.shape()[0] == 1 {
		if self.index <= self.rt.shape()[1] {
		    Some(self.index)
		}
		else {
		    None
		}
	    }
	    else {
		if self.index < self.rt.shape()[0] {
		    Some(self.index)
		}
		else {
		    None
		}
	    }
	}
	else {
	    if self.index < self.rt.shape()[0] {
		Some(self.index)
	    }
	    else {
		None
	    }
	};
	match current {
	    Some(current) => {
		self.index += 1;
		Some(self.rt.subtensor(current))
	    }
	    None => None
	}
    }
}
