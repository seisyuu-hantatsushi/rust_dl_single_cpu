use std::{ops};
use num;

use crate::tensor::tensor_base::{Tensor};

/* T = T - T */
impl<T> ops::Sub for Tensor<T>
   where T:num::Num + Copy
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) - (*rhs)).collect::<Vec<T>>();

	Tensor::from_vector(self.shape().to_owned(), v)
    }
}

/* T = &T - T */
impl<T> ops::Sub<Tensor<T>> for &Tensor<T>
   where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn sub(self, other: Tensor<T>) -> Self::Output {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) - (*rhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

/* T = T - &T */
impl<T> ops::Sub<&Tensor<T>> for Tensor<T>
   where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn sub(self, other: &Self) -> Self::Output {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) - (*rhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

/* T = &T + &T */
impl<T> ops::Sub for &Tensor<T>
   where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn sub(self, other: Self) -> Self::Output {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) - (*rhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

#[test]
fn tensor_sub_test(){
    let shape:[usize;2] = [3,4];
    let t0 = Tensor::<f32>::zero(&shape);
    let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
			    21.0,22.0,23.0,24.0,
			    31.0,32.0,33.0,34.0 ];
    let t1 = Tensor::<f32>::from_array(&[3,4],&m_init);

    let t2 = t0 - t1;
    println!("{}", t2);
}
