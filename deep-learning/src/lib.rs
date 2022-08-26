pub mod activator;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;
    use activator::*;
    use linear_transform::matrix::{MatrixMxN,matrix_mxn};

    #[test]
    fn activator_test() {
	let v1 = MatrixMxN::from_f64(1, 3, &[-1.0,1.0,2.0]);
	let v2 = sigmoid(&v1);
	let v3 = MatrixMxN::from_f64(1, 3, &[1.0/(1.0+f64::consts::E.powf(1.0)),1.0/(1.0+f64::consts::E.powf(-1.0)),1.0/(1.0+f64::consts::E.powf(-2.0))]);
	assert_eq!(v2,v3);

	let v4 = relu(&v1);
	assert_eq!(v4, MatrixMxN::from_f64(1, 3, &[0.0, 1.0, 2.0]));
    }

    #[test]
    fn network_test() {
	let input_data:Vec<Vec<f64>> = vec![vec![1.0, 0.5]];
	let w1_data:Vec<Vec<f64>>    = vec![vec![0.1, 0.3, 0.5], vec![0.2, 0.4, 0.6]];
	let b1_data:Vec<Vec<f64>>    = vec![vec![0.1, 0.2, 0.3]];
	let w2_data:Vec<Vec<f64>>    = vec![vec![0.1, 0.4], vec![0.2, 0.5], vec![0.3, 0.6]];
	let b2_data:Vec<Vec<f64>>    = vec![vec![0.1, 0.2]];
	let w3_data:Vec<Vec<f64>>    = vec![vec![0.1, 0.3], vec![0.2, 0.4]];
	let b3_data:Vec<Vec<f64>>    = vec![vec![0.1, 0.2]];

	let x  = MatrixMxN::from_vector_f64(input_data);
	let (w1,w2,w3) = (MatrixMxN::from_vector_f64(w1_data),MatrixMxN::from_vector_f64(w2_data),MatrixMxN::from_vector_f64(w3_data));
	let (b1,b2,b3) = (MatrixMxN::from_vector_f64(b1_data),MatrixMxN::from_vector_f64(b2_data),MatrixMxN::from_vector_f64(b3_data));
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
}
