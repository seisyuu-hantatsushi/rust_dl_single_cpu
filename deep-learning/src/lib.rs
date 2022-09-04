pub mod activator;
pub mod output_layer;
pub mod loss_functions;
pub mod gradient;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32,f64};
    use activator::*;
    use output_layer::*;
    use loss_functions::*;
    use linear_transform::matrix::{MatrixMxN,matrix_mxn};

    #[test]
    fn activator_test() {
	let v1 = MatrixMxN::from_array(1, 3, &[-1.0,1.0,2.0]);
	let v2 = sigmoid(&v1);
	let v3 = MatrixMxN::from_array(1, 3, &[1.0/(1.0+f64::consts::E.powf(1.0)),1.0/(1.0+f64::consts::E.powf(-1.0)),1.0/(1.0+f64::consts::E.powf(-2.0))]);
	assert_eq!(v2,v3);

	let v4 = relu(&v1);
	assert_eq!(v4, MatrixMxN::from_array(1, 3, &[0.0, 1.0, 2.0]));
    }

    #[test]
    fn network_test() {
	{
	    let input_data:Vec<Vec<f32>> = vec![vec![1.0, 0.5]];
	    let w1_data:Vec<Vec<f32>>    = vec![vec![0.1, 0.3, 0.5], vec![0.2, 0.4, 0.6]];
	    let b1_data:Vec<Vec<f32>>    = vec![vec![0.1, 0.2, 0.3]];
	    let w2_data:Vec<Vec<f32>>    = vec![vec![0.1, 0.4], vec![0.2, 0.5], vec![0.3, 0.6]];
	    let b2_data:Vec<Vec<f32>>    = vec![vec![0.1, 0.2]];
	    let w3_data:Vec<Vec<f32>>    = vec![vec![0.1, 0.3], vec![0.2, 0.4]];
	    let b3_data:Vec<Vec<f32>>    = vec![vec![0.1, 0.2]];

	    let x  = MatrixMxN::from_vector(input_data);
	    let (w1,w2,w3) = (MatrixMxN::from_vector(w1_data),MatrixMxN::from_vector(w2_data),MatrixMxN::from_vector(w3_data));
	    let (b1,b2,b3) = (MatrixMxN::from_vector(b1_data),MatrixMxN::from_vector(b2_data),MatrixMxN::from_vector(b3_data));
	    println!("{}", x);
	    println!("{}", w2);

	    let a1 = x * w1 + b1;
	    let z1 = sigmoid(&a1);
	    let a2 = z1 * w2 + b2;
	    let z2 = sigmoid(&a2);
	    let a3 = z2 * w3 + b3;
	    let y = matrix_mxn::identity(a3);

	    println!("{}", y);
	    //y = [0.31682707641102975,0.6962790898619668]
	}
	{
	    let input_data:Vec<Vec<f32>> = vec![vec![0.3, 2.9, 4.0]];
	    let a = MatrixMxN::from_vector(input_data);
	    let y = softmax(&a);
	    let s = y.into_vector().into_iter().reduce(|x,y| x+y).unwrap();
	    println!("{}", s);
	    assert!((1.0-s).abs() < 1e-6);
	}
   }
    #[test]
    fn loss_functions_test() {
	let t = MatrixMxN::<f32>::from_array(1, 10, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);
	let e = sum_squarted_error(&y,&t);
	assert!((0.0975-e).abs() < 1e-6);

	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
	let e = sum_squarted_error(&y,&t);
	assert!((0.5975-e).abs() < 1e-6);

	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);
	let e = cross_entropy_error(&y,&t);
	assert!((0.510825457-e).abs() < 1e-6);

	let y = MatrixMxN::<f32>::from_array(1, 10, &[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
	let e = cross_entropy_error(&y,&t);
	assert!((2.302584092-e).abs() < 1e-6);
    }
}
