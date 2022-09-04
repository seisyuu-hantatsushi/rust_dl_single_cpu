use std::ops;
use num::{Float,FromPrimitive};
use linear_transform::matrix::MatrixMxN;

pub fn sum_squarted_error<T:Float+ops::AddAssign+ops::MulAssign>(y: &MatrixMxN<T>, t: &MatrixMxN<T>) -> T {
    assert_eq!(y.shape(), t.shape());
    let two = num::one::<T>()+num::one::<T>();
    let sum:T = y.buffer().iter().zip(t.buffer().iter()).fold(num::zero(), |sum, (y,t)| { let two = num::one::<T>()+num::one::<T>(); sum+(*y-*t).powf(two) });
    sum/two
}

pub fn cross_entropy_error<T:Float+FromPrimitive+ops::AddAssign+ops::MulAssign>(y: &MatrixMxN<T>, t: &MatrixMxN<T>) -> T {
    assert_eq!(y.shape(), t.shape());
    let sum:T = y.buffer().iter().zip(t.buffer().iter()).fold(num::zero(), |sum, (y,t)| {
	let delta:T = FromPrimitive::from_f64(1.0e-7).unwrap();
	sum+(*t)*((*y+delta).ln())
    });
    sum.neg()
}
