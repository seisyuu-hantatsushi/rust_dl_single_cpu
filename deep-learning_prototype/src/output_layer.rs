
use std::{ops};
use num;
use linear_transform::matrix::MatrixMxN;
use linear_transform::tensor::tensor_base::Tensor;

pub trait Softmax<T> {
    fn softmax(&self) -> T;
}

impl<T> Softmax<MatrixMxN<T>> for MatrixMxN<T>
    where T: num::Float+ops::AddAssign+ops::MulAssign
{
    fn softmax(&self) -> MatrixMxN<T> {
	let (c, r) = self.shape();
	let one:T = num::one();
	let e = one.exp();
	let max_of_elements =
	    self.buffer().iter().reduce(|x,y| if x < y { y } else { x }).unwrap();
	let v:Vec<T> = self.buffer().iter().map(|x| e.powf(*x - *max_of_elements)).collect();
	let s = v.clone().into_iter().reduce(|x,y| x+y).unwrap();
	let sv:Vec<T> = v.iter().map(|x| (*x)/s).collect();
	MatrixMxN::from_array(c, r, &sv.into_boxed_slice())
    }
}

impl<T> Softmax<Tensor<T>> for Tensor<T>
    where T: num::Float
{
    fn softmax(&self) -> Tensor<T> {
	let one:T = num::one();
	let e = one.exp();
	let max_of_elements =
	    self.buffer().iter().reduce(|x,y| if x < y { y } else { x }).unwrap();
	let v:Vec<T> = self.buffer().iter().map(|x| e.powf(*x - *max_of_elements)).collect();
	let s = v.clone().into_iter().reduce(|x,y| x+y).unwrap();
	let sv:Vec<T> = v.iter().map(|x| (*x)/s).collect();
	Tensor::from_array(self.shape(), &sv.into_boxed_slice())
    }
}
