/* -*- tab-width:4 -*- */
use std::ops;
use num;
use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

    pub fn mul_rank0(lhs:&Tensor<T>, rhs:&Tensor<T>) -> Tensor<T> {
	assert_eq!(lhs.shape(), &[1,1]);
	assert_eq!(rhs.shape(), &[1,1]);
	Tensor::<T>::from_array(&[1,1], &[lhs[vec![0,0]]*rhs[vec![0,0]]])
    }

    pub fn hadamard_product(lhs:&Tensor<T>, rhs:&Tensor<T>) -> Tensor<T> {
	assert_eq!(lhs.shape(), rhs.shape());
	let v = lhs.buffer().iter().zip(rhs.buffer().iter()).map(|(&l, &r)| { l*r }).collect::<Vec<T>>();
	Tensor::from_vector(lhs.shape().to_vec(), v)
    }
}

impl<T> ops::Mul for Tensor<T>
where T:num::Num + Copy
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
	match (self.shape().len(), other.shape().len()) {
	    (0,0) => {
		Tensor::<T>::from_array(&[], &[self.buffer()[0] * other.buffer()[0]])
	    },
	    (0,1) => {
		let v = other.buffer().iter().map(|&e| self.buffer()[0]*e).collect::<Vec<T>>();
		Tensor::<T>::from_vector(other.shape().to_vec(), v)
	    },
	    (1,0) => {
		let v = self.buffer().iter().map(|&e| other.buffer()[0]*e).collect::<Vec<T>>();
		Tensor::<T>::from_vector(self.shape().to_vec(), v)
	    },
	    (_,_) => {
		Tensor::<T>::zero(&[])
	    }
	}
    }
}

impl<T> ops::Mul for &Tensor<T>
where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn mul(self, other: Self) -> Self::Output {
	match (self.shape().len(), other.shape().len()) {
	    (0,0) => {
		Tensor::<T>::from_array(&[], &[self.buffer()[0] * other.buffer()[0]])
	    },
	    (0,1) => {
		let v = other.buffer().iter().map(|&e| self.buffer()[0]*e).collect::<Vec<T>>();
		Tensor::<T>::from_vector(other.shape().to_vec(), v)
	    },
	    (1,0) => {
		let v = self.buffer().iter().map(|&e| other.buffer()[0]*e).collect::<Vec<T>>();
		Tensor::<T>::from_vector(self.shape().to_vec(), v)
	    },
	    (_,_) => {
		Tensor::<T>::zero(&[])
	    }
	}
    }
}

impl<T> ops::Mul<&Tensor<T>> for Tensor<T>
where T:num::Num + Copy {
    type Output = Tensor<T>;
    fn mul(self, other: &Self) -> Self::Output {
	match (self.shape().len(), other.shape().len()) {
	    (0,0) => {
		Tensor::<T>::from_array(&[], &[self.buffer()[0] * other.buffer()[0]])
	    },
	    (0,1) => {
		let v = other.buffer().iter().map(|&e| self.buffer()[0]*e).collect::<Vec<T>>();
		Tensor::<T>::from_vector(other.shape().to_vec(), v)
	    },
	    (1,0) => {
		let v = self.buffer().iter().map(|&e| other.buffer()[0]*e).collect::<Vec<T>>();
		Tensor::<T>::from_vector(self.shape().to_vec(), v)
	    },
	    (_,_) => {
		Tensor::<T>::zero(&[])
	    }
	}
    }
}
