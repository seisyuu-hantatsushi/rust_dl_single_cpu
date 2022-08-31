use std::{ops};
use num::Float;
use linear_transform::matrix::MatrixMxN;

pub fn sigmoid<T: Float+ops::AddAssign+ops::MulAssign>(m: &MatrixMxN<T>) -> MatrixMxN<T> {
    let (c,r) = m.shape();
    let one:T = num::one();
    let e = one.exp();
    let minus_one = one.neg();
    let b:Vec<T> = m.buffer().iter().map(|v| one/(one+e.powf(minus_one*(*v)))).collect();
    MatrixMxN::<T>::from_array(c,r,&b.into_boxed_slice())
}

pub fn relu<T: num::Float+ops::AddAssign+ops::MulAssign>(m: &MatrixMxN<T>) -> MatrixMxN<T> {
    let (c,r) = m.shape();
    let b:Vec<T> = m.buffer().iter().map(|v| if *v > num::zero() {*v} else {num::zero()}).collect();
    MatrixMxN::from_array(c, r, &b.into_boxed_slice())
}
