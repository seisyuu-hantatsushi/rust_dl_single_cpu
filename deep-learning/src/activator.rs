use std::{f64};
use linear_transform::matrix::MatrixMxN;

pub fn sigmoid(m: &MatrixMxN) -> MatrixMxN {
    let (c,r) = m.shape();
    let b:Vec<f64> = m.iter().map(|v| 1.0/(1.0+f64::consts::E.powf(-v))).collect();
    MatrixMxN::from_f64(c, r, &b.into_boxed_slice())
}

pub fn relu(m: &MatrixMxN) -> MatrixMxN {
    let (c,r) = m.shape();
    let b:Vec<f64> = m.iter().map(|v| if *v > 0.0 {*v} else {0.0}).collect();
    MatrixMxN::from_f64(c, r, &b)
}
