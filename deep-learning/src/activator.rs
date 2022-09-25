use std::ops;
use num::Float;
use linear_transform::matrix::MatrixMxN;
use linear_transform::tensor::tensor_base::Tensor;

pub trait Activator<T> {
    fn sigmoid(&self) -> T;
    fn relu(&self) -> T;
}

impl<T> Activator<Tensor<T>> for Tensor<T>
where T:num::Float+Clone {
    fn sigmoid(&self) -> Tensor<T> {
	let one:T = num::one();
	let e = one.exp();
	let minus_one = one.neg();
	let b:Vec<T> = self.buffer().iter().map(|v| one/(one+e.powf(minus_one*(*v)))).collect();
	Tensor::from_vector(self.shape().to_vec(), b)
    }

    fn relu(&self) -> Tensor<T> {
	let b:Vec<T> = self.buffer().iter().map(|v| if *v > num::zero() {*v} else {num::zero()}).collect();
	Tensor::from_vector(self.shape().to_vec(), b)
    }
}

impl<T> Activator<MatrixMxN<T>> for MatrixMxN<T>
    where T: num::Float+ops::AddAssign+ops::MulAssign {

    fn sigmoid(&self) -> MatrixMxN<T> {
	let (c,r) = self.shape();
	let one:T = num::one();
	let e = one.exp();
	let minus_one = one.neg();
	let b:Vec<T> = self.buffer().iter().map(|v| one/(one+e.powf(minus_one*(*v)))).collect();
	MatrixMxN::<T>::from_array(c,r,&b.into_boxed_slice())
    }

    fn relu(&self) -> MatrixMxN<T> {
	let (c,r) = self.shape();
	let b:Vec<T> = self.buffer().iter().map(|v| if *v > num::zero() {*v} else {num::zero()}).collect();
	MatrixMxN::from_array(c, r, &b.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32,f64};
    #[test]
    fn activator_tensor_test () {
	let v1 = Tensor::<f64>::from_array(&[1, 3], &[-1.0,1.0,2.0]);
	let v2 = v1.sigmoid();
	let v3 = Tensor::<f64>::from_array(&[1, 3], &[1.0/(1.0+f64::consts::E.powf(1.0)),1.0/(1.0+f64::consts::E.powf(-1.0)),1.0/(1.0+f64::consts::E.powf(-2.0))]);
	assert_eq!(v2,v3);

	let v4 = Tensor::<f64>::sigmoid(&v1);
	assert_eq!(v2,v4);

	let v5 = v1.relu();
	assert_eq!(v5, Tensor::<f64>::from_array(&[1, 3], &[0.0, 1.0, 2.0]));
    }

    #[test]
    fn activator_matrix_test() {
	let v1 = MatrixMxN::from_array(1, 3, &[-1.0,1.0,2.0]);
	let v2 = MatrixMxN::sigmoid(&v1);
	let v3 = MatrixMxN::from_array(1, 3, &[1.0/(1.0+f64::consts::E.powf(1.0)),1.0/(1.0+f64::consts::E.powf(-1.0)),1.0/(1.0+f64::consts::E.powf(-2.0))]);
	assert_eq!(v2,v3);

	let v4 = v1.relu();
	assert_eq!(v4, MatrixMxN::from_array(1, 3, &[0.0, 1.0, 2.0]));
    }
}
