
use num;

use crate::tensor::tensor_base::{Tensor};
//use crate::tensor::tensor_add::*;

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

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
	for i in 0..m {
	    let a = lhs.sub_tensor(i);
	    for j in 0..l {
		let b = rhs_transpose.sub_tensor(j);
		v[i*l+j] = a.buffer().iter().zip(b.buffer().iter()).fold(num::zero(),|sum,(ai,bi)| sum+(*ai)*(*bi));
	    }
	}
	Tensor::from_vector(vec![m,l], v)
    }

    pub fn affine(a:&Tensor<T>, v:&Tensor<T>, b:&Tensor<T>) -> Tensor<T> {
	Tensor::matrix_product(a,v) + b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_affine_test() {
	let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
				21.0,22.0,23.0,24.0,
				31.0,32.0,33.0,34.0 ];
	let t1 = Tensor::<f32>::from_array(&[3,4],&m_init);
	let m_init:[f32;20] = [ 11.0,12.0,13.0,14.0,15.0,
				21.0,22.0,23.0,24.0,25.0,
				31.0,32.0,33.0,34.0,35.0,
				41.0,42.0,43.0,44.0,45.0 ];
	let t2 = Tensor::<f32>::from_array(&[4,5],&m_init);
	let t3 = Tensor::<f32>::matrix_product(&t1, &t2);
	assert_eq!(t3[vec![0,0]], 11.0*11.0+12.0*21.0+13.0*31.0+14.0*41.0);
	assert_eq!(t3[vec![1,0]], 21.0*11.0+22.0*21.0+23.0*31.0+24.0*41.0);
	assert_eq!(t3[vec![0,1]], 11.0*12.0+12.0*22.0+13.0*32.0+14.0*42.0);
	
	let m_init:[f32;9] = [ 2.0,0.0,1.0,
			       2.0,2.0,1.0,
			       3.0,0.0,1.0 ];
	let a = Tensor::<f32>::from_array(&[3,3],&m_init);
	let m_init:[f32;3] = [ 1.0, 2.0, 3.0 ];
	let v = Tensor::<f32>::from_array(&[3,1],&m_init);
	let m_init:[f32;3] = [ 2.0, 2.0, 2.0 ];
	let b = Tensor::<f32>::from_array(&[3,1],&m_init);
	let c = Tensor::<f32>::affine(&a,&v,&b);
	assert_eq!(c[vec![1,0]],2.0*1.0+2.0*2.0+1.0*3.0+2.0);
    }
}
