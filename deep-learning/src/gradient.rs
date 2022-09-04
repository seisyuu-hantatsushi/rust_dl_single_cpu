use std::ops;
use num::{Float,FromPrimitive};
use linear_transform::matrix::MatrixMxN;

fn gradient<T:Float,F:fn (m:&MatrixMxN<T>>) -> T>(f:F, p:&MatrixMxN<T>) -> MatrixMxN<T> {
    
    return num::one();
}
