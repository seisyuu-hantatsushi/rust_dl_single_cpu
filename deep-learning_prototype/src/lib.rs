pub mod activator;
pub mod output_layer;
pub mod loss_functions;
pub mod gradient;
pub mod loader;
pub mod neural_network;
pub mod neuron;
pub mod synapse;

#[cfg(_test)]
mod tests {
    use super::*;
    use std::f32;
    use linear_transform::matrix::{MatrixMxN,matrix_mxn};
    use activator::*;
    use output_layer::*;

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
	    let z1 = a1.sigmoid();
	    let a2 = z1 * w2 + b2;
	    let z2 = a2.sigmoid();
	    let a3 = z2 * w3 + b3;
	    let y = matrix_mxn::identity(a3);

	    println!("{}", y);
	    //y = [0.31682707641102975,0.6962790898619668]
	}
	{
	    let input_data:Vec<Vec<f32>> = vec![vec![0.3, 2.9, 4.0]];
	    let a = MatrixMxN::from_vector(input_data);
	    let y = MatrixMxN::<f32>::softmax(&a);
	    let s = y.into_vector().into_iter().reduce(|x,y| x+y).unwrap();
	    println!("{}", s);
	    assert!((1.0-s).abs() < 1e-6);
	}
   }
}
