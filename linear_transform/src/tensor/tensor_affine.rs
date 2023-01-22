
use num;

use crate::tensor::tensor_base::Tensor;
//use crate::tensor::tensor_add::*;

impl<T> Tensor<T>
where T:num::Num+Clone+Copy+std::fmt::Display {

    pub fn transpose(&self) -> Tensor<T> {
	assert_eq!(self.shape().len(),2);
	let (c, r) = (self.shape()[0], self.shape()[1]);
	let mut v:Vec<T> = Vec::with_capacity(c*r);
	unsafe { v.set_len(c*r) };
	let src = self.buffer();
	for i in 0..c {
	    for j in 0..r {
		v[j*c+i] = src[i*r+j];
	    }
	}
	Tensor::from_vector(vec![r,c], v)
    }

    pub fn matrix_product(lhs:&Tensor<T>,rhs:&Tensor<T>) -> Tensor<T> {
	// until Rank 3 Tensor
	assert_eq!(lhs.shape().len(),2);
	assert_eq!(rhs.shape().len(),2);
	assert_eq!(lhs.shape()[1], rhs.shape()[0]);
	let (m,l) = (lhs.shape()[0],rhs.shape()[1]);
	let dst_size = lhs.shape()[0]*rhs.shape()[1];
	let mut v:Vec<T> = Vec::with_capacity(dst_size);
	unsafe { v.set_len(dst_size) };
	let rhs_transpose = rhs.transpose();
	let rhs_tranpose_shape = rhs_transpose.shape().to_vec();
	for i in 0..m {
	    let a = if lhs.shape()[0] == 1 {
		lhs.subtensor_all()
	    }
	    else {
		lhs.subtensor(i)
	    };
	    for j in 0..l {
		let b = if rhs_tranpose_shape[0] == 1 {
		    rhs_transpose.subtensor_all()
		}
		else {
		    rhs_transpose.subtensor(j)
		};
		v[i*l+j] = a.buffer().iter().zip(b.buffer().iter()).fold(num::zero(),|sum,(ai,bi)| sum+(*ai)*(*bi));
	    }
	}
	Tensor::from_vector(vec![m,l], v)
    }

    pub fn affine(a:&Tensor<T>, v:&Tensor<T>, b:&Tensor<T>) -> Tensor<T> {
	Tensor::matrix_product(a,v) + b
    }
}
