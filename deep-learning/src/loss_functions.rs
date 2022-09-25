use std::ops;
use num::FromPrimitive;
use linear_transform::matrix::MatrixMxN;
use linear_transform::tensor::tensor_base::Tensor;

pub trait LossFunction<T,S> {
    fn sum_squarted_error(y:&T, t:&T) -> S;
    fn cross_entropy_error(y:&T, t:&T) -> S;
}

impl<T> LossFunction<MatrixMxN<T>,T> for MatrixMxN<T>
where T:num::Float+num::FromPrimitive+ops::AddAssign+ops::MulAssign
{
    fn sum_squarted_error(y: &MatrixMxN<T>, t: &MatrixMxN<T>) -> T {
	assert_eq!(y.shape(), t.shape());
	let two = num::one::<T>()+num::one::<T>();
	let sum:T = y.buffer().iter().zip(t.buffer().iter()).fold(num::zero(), |sum, (y,t)| { let two = num::one::<T>()+num::one::<T>(); sum+(*y-*t).powf(two) });
	sum/two
    }

    fn cross_entropy_error(y: &MatrixMxN<T>, t: &MatrixMxN<T>) -> T {
	assert_eq!(y.shape(), t.shape());
	let sum:T = y.buffer().iter().zip(t.buffer().iter()).fold(num::zero(), |sum, (y,t)| {
	    let delta:T = FromPrimitive::from_f64(1.0e-7).unwrap();
	    sum+(*t)*((*y+delta).ln())
	});
	sum.neg()
    }
}

impl<T> LossFunction<Tensor<T>,T> for Tensor<T>
where T:num::Float+num::FromPrimitive {
    fn sum_squarted_error(y: &Tensor<T>, t: &Tensor<T>) -> T {
	assert_eq!(y.shape(), t.shape());
	let two = num::one::<T>()+num::one::<T>();
	let sum:T = y.buffer().iter().zip(t.buffer().iter()).fold(num::zero(), |sum, (y,t)| { let two = num::one::<T>()+num::one::<T>(); sum+(*y-*t).powf(two) });
	sum/two
    }

    fn cross_entropy_error(y: &Tensor<T>, t: &Tensor<T>) -> T {
	assert_eq!(y.shape(), t.shape());
	let sum:T = y.buffer().iter().zip(t.buffer().iter()).fold(num::zero(), |sum, (y,t)| {
	    let delta:T = FromPrimitive::from_f64(1.0e-7).unwrap();
	    sum+(*t)*((*y+delta).ln())
	});
	sum.neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32;
    #[test]
    fn loss_functions_matrix_test() {
	let t = MatrixMxN::<f32>::from_array(1, 10, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);
	let e = MatrixMxN::<f32>::sum_squarted_error(&y,&t);
	assert!((0.0975-e).abs() < 1e-6);

	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
	let e = MatrixMxN::<f32>::sum_squarted_error(&y,&t);
	assert!((0.5975-e).abs() < 1e-6);

	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);
	let e = MatrixMxN::<f32>::cross_entropy_error(&y,&t);
	assert!((0.510825457-e).abs() < 1e-6);

	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
	let e = MatrixMxN::<f32>::cross_entropy_error(&y,&t);
	assert!((2.302584092-e).abs() < 1e-6);
    }

    #[test]
    fn loss_functions_tensor_test() {
	let t = Tensor::<f32>::from_array(&[1, 10], &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let y = Tensor::<f32>::from_array(&[1, 10], &[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);
	let e = Tensor::<f32>::sum_squarted_error(&y,&t);
	assert!((0.0975-e).abs() < 1e-6);

	let y = Tensor::<f32>::from_array(&[1, 10], &[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
	let e = Tensor::<f32>::sum_squarted_error(&y,&t);
	assert!((0.5975-e).abs() < 1e-6);

	let y = Tensor::<f32>::from_array(&[1, 10], &[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);
	let e = Tensor::<f32>::cross_entropy_error(&y,&t);
	assert!((0.510825457-e).abs() < 1e-6);

	let y = Tensor::<f32>::from_array(&[1, 10], &[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
	let e = Tensor::<f32>::cross_entropy_error(&y,&t);
	assert!((2.302584092-e).abs() < 1e-6);
    }
}
