use std::{ops};
use num;

use crate::tensor::tensor_base::{Tensor,SubTensor};

/* T = T + T */
impl<T> ops::Add for Tensor<T>
   where T:num::Num + Copy
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) + (*rhs)).collect::<Vec<T>>();

	Tensor::from_vector(self.shape().to_owned(), v)
    }
}

/* T = &T + T */
impl<T> ops::Add<Tensor<T>> for &Tensor<T>
   where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn add(self, other: Tensor<T>) -> Self::Output {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) + (*rhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

/* T = T + &T */
impl<T> ops::Add<&Tensor<T>> for Tensor<T>
   where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn add(self, other: &Self) -> Self::Output {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) + (*rhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

/* T = &T + &T */
impl<T> ops::Add for &Tensor<T>
   where T:num::Num + Copy
{
    type Output = Tensor<T>;
    fn add(self, other: Self) -> Self::Output {
	assert_eq!(self.shape(),other.shape());
	let lhs:&[T] = self.buffer();
	let rhs:&[T] = other.buffer();
	let v = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| (*lhs) + (*rhs)).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }
}

impl<T> Tensor<T>
where T:num::Num + Copy + std::fmt::Display{

    pub fn add_at(&self, pos:&[usize], x:&Tensor<T>) -> Tensor<T> {
	assert!(pos.len() > 0);
	let st = self.get_sub_tensor_by_position(pos).unwrap_or_else(|| panic!("invalid position"));
	let start_pos = self.position_to_index(pos).unwrap_or_else(|| panic!("invalid position"));
	let mut v = self.buffer().to_vec();
	let xv = x.buffer();
	for i in 0..st.num_of_elements() {
	    v[start_pos+i] = v[start_pos+i]+xv[i]
	}
	Tensor::from_vector(self.shape().to_vec(), v)
   }
}

