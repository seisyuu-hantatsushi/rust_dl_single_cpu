use std::ops;
use num::{Float,FromPrimitive};
use linear_transform::matrix::MatrixMxN;

fn gradient<T,F>(f:F, p:&MatrixMxN<T>) -> MatrixMxN<T>
where T:Float+FromPrimitive+ops::AddAssign+ops::MulAssign,
      F:Fn(&MatrixMxN<T>) -> T
{
    let (c, r) = p.shape();
    let delta:T = FromPrimitive::from_f64(5.0e-5).unwrap();
    let two:T = FromPrimitive::from_f64(2.0).unwrap();
    let ref_buffer = p.buffer();
    let mut g = Vec::<T>::with_capacity(ref_buffer.len());
    unsafe{ g.set_len(ref_buffer.len()); }
    for i in 0..ref_buffer.len() {
	let front_v =
	    (0..ref_buffer.len()).zip(ref_buffer.iter()).
	    map(|(idx, v)| if idx == i {*v+delta} else {*v}).collect::<Vec<T>>();
	let f_front = f(&MatrixMxN::from_array(c, r, &front_v.into_boxed_slice()));
	let back_v =
	    (0..ref_buffer.len()).zip(ref_buffer.iter()).
	    map(|(idx, v)| if idx == i {*v-delta} else {*v}).collect::<Vec<T>>();
	let f_back = f(&MatrixMxN::from_array(c, r, &back_v.into_boxed_slice()));
	g[i] = (f_front - f_back)/(two*delta);
    }
    return MatrixMxN::from_array(c, r, &g.into_boxed_slice());
}

pub fn gradient_descent<F,T>(f:F, init:MatrixMxN<T>, learning_rate:T, number_of_step:u32) -> MatrixMxN<T>
where T:Float+FromPrimitive+ops::AddAssign+ops::MulAssign,
      F:Fn(&MatrixMxN<T>) -> T {
    let mut m = init.clone();
    let mut i = 0;
    while i < number_of_step {
	let grad = gradient(&f, &m);
	m = m-grad*learning_rate;
	i += 1;
    }
    m
}

#[test]
fn gradiant_test() {
    let f2 = |m: &MatrixMxN<f32>| -> f32 {
	let (c, r) = m.shape();
	assert!(c == 1 && r == 2);
	m[0][0].powf(2.0) + m[0][1].powf(2.0)
    };
    let p:Vec<Vec<f32>> = vec![vec![3.0f32, 4.0f32]];
    let g  = gradient(f2, &MatrixMxN::<f32>::from_vector(p));
    //println!("{} ", (((3.0f32+0.00005f32).powf(2.0)+16.0f32)-((3.0f32-0.00005f32).powf(2.0)+16.0f32))/0.0001);
    //println!("{} ", (((3.0f64+0.0001f64).powf(2.0)+16.0f64)-((3.0f64-0.0001f64).powf(2.0)+16.0f64))/0.0002);
    //println!("{} {}",g[0][0],g[0][1]);
    assert!((g[0][0]-6.0).abs() < 1e-1);
    assert!((g[0][1]-8.0).abs() < 1e-1);
    let p:Vec<Vec<f32>> = vec![vec![0.0f32, 2.0f32]];
    let g = gradient(f2, &MatrixMxN::<f32>::from_vector(p));
    assert!((g[0][0]-0.0).abs() < 1e-1);
    assert!((g[0][1]-4.0).abs() < 1e-1);
    let p:Vec<Vec<f32>> = vec![vec![3.0f32, 0.0f32]];
    let g = gradient(f2, &MatrixMxN::<f32>::from_vector(p));
    assert!((g[0][0]-6.0).abs() < 1e-1);
    assert!((g[0][1]-0.0).abs() < 1e-1);

    let init = MatrixMxN::<f32>::from_vector(vec![vec![-3.0f32, 4.0f32]]);
    let result = gradient_descent(f2, init, 0.1f32, 100);

    assert!((result[0][0]-0.0).abs() < 1e-8);
    assert!((result[0][1]-0.0).abs() < 1e-8);
}
