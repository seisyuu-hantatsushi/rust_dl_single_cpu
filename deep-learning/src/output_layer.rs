
use std::{ops};
use num;
use linear_transform::matrix::MatrixMxN;

pub fn softmax<T: num::Float+ops::AddAssign+ops::MulAssign>(m: &MatrixMxN<T>) -> MatrixMxN<T> {
    let (c, r) = m.shape();
    let one:T = num::one();
    let e = one.exp();
    let max_of_elements =
	m.buffer().iter().reduce(|x,y| if x < y { y } else { x }).unwrap();
    let v:Vec<T> = m.buffer().iter().map(|x| e.powf(*x - *max_of_elements)).collect();
    let s = v.clone().into_iter().reduce(|x,y| x+y).unwrap();
    let sv:Vec<T> = v.iter().map(|x| (*x)/s).collect();

    MatrixMxN::from_array(c, r, &sv.into_boxed_slice())
}
