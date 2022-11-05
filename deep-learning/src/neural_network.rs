/* -*- tab-width:4 -*- */
use std::fmt;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use std::collections::HashMap;
use linear_transform::tensor::tensor_base::Tensor;

use crate::neuron::{NNNeuron,Neuron,nn_neuron_new};
use crate::synapse::{NNSynapseNode,SynapseNode};

struct NeuralNetwork<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	neurons: HashMap<* const RefCell<Neuron<T>>, NNNeuron<T>>,
	synapse_node: HashMap<* const RefCell<SynapseNode<T>>, Rc<RefCell<SynapseNode<T>>>>,
	generation_table: Vec<NNSynapseNode<T>>
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn nn_test() {
		{
			let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[3.0]));
			let (add1,output) = SynapseNode::<f64>::add(Rc::clone(&x),Rc::clone(&x));
			let (add2,output) = SynapseNode::<f64>::add(Rc::clone(&output),Rc::clone(&x));
			add2.borrow().make_diff_node();
			let sns = add1.borrow().make_diff_node();
			sns[0].borrow().forward();
			let gx = sns[1].borrow().forward();
			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[9.0]));
			assert_eq!(gx[0].borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[3.0]));
		}

		{
			let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[2.0]));
			let (neg, output) = SynapseNode::<f64>::neg(Rc::clone(&x));
			let (sub1,output) = SynapseNode::<f64>::sub(output,Rc::clone(&x));
			let (sub2,output) = SynapseNode::<f64>::sub(output,Rc::clone(&x));
			sub2.borrow().make_diff_node();
			sub1.borrow().make_diff_node();
			neg.borrow().make_diff_node();

			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-6.0]));

			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-3.0]));
			}
			else {
				println!("x.grad=None");
				assert!(false);
			}

			let y = nn_neuron_new::<f64>("y", Tensor::<f64>::from_array(&[1,1],&[2.0]));
			let (neg, neg_y) = SynapseNode::<f64>::neg(Rc::clone(&y));
			let (add1,output) = SynapseNode::<f64>::add(Rc::clone(&neg_y),Rc::clone(&neg_y));
			let (add2,output) = SynapseNode::<f64>::add(output,Rc::clone(&neg_y));
			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-6.0]));
			add2.borrow().make_diff_node();
			add1.borrow().make_diff_node();
			neg.borrow().make_diff_node();
			let borrowed_y = y.borrow();
			if let Some(ref g) = borrowed_y.ref_grad() {
				assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-3.0]));
			}
			else {
				println!("y.grad=None");
				assert!(false);
			}
		}

		{
			let a = nn_neuron_new::<f64>("a", Tensor::<f64>::from_array(&[1,1],&[3.0]));
			let b = nn_neuron_new::<f64>("b", Tensor::<f64>::from_array(&[1,1],&[2.0]));
			let c = nn_neuron_new::<f64>("c", Tensor::<f64>::from_array(&[1,1],&[1.0]));
			let (mul1,output) = SynapseNode::<f64>::mul_rank0(Rc::clone(&a),Rc::clone(&b));
			let (add1,output) = SynapseNode::<f64>::add(output,c);
			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[7.0]));
			add1.borrow().make_diff_node();
			mul1.borrow().make_diff_node();

			{
				let borrowed_a = a.borrow();
				if let Some(ref g) = borrowed_a.ref_grad() {
					assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[2.0]));
				}
				else {
					println!("a.grad=None");
					assert!(false);
				}
			}
			{
				let borrowed_b = b.borrow();
				if let Some(ref g) = borrowed_b.ref_grad() {
					assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[3.0]));
				}
				else {
					println!("b.grad=None");
					assert!(false);
				}
			}
		}

		{
			let x = 3.0;
			let a = nn_neuron_new::<f64>("a", Tensor::<f64>::from_array(&[1,1],&[x]));
			let (mul1,output) = SynapseNode::<f64>::mul_rank0(Rc::clone(&a),Rc::clone(&a));
			/* z = a*a */
			println!("{}",output.borrow());
			mul1.borrow().make_diff_node();
			{
				let borrowed_a = a.borrow();
				if let Some(ref g) = borrowed_a.ref_grad() {
					/*
					fn square(x:f64) -> f64 {
						x*x
					}
					let delta = 1.0e-4;
					let diff = (square(x+delta)-square(x-delta))/(2.0*delta);
					println!("{}", diff);
					 */
					assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[6.0]));
				}
				else {
					println!("a.grad=None");
					assert!(false);
				}
			}

			{
				let x0 = 0.5;
				let x = nn_neuron_new::<f64>("a", Tensor::<f64>::from_array(&[1,1],&[x0]));
				let two =  nn_neuron_new::<f64>("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
				let (square_sn, a) = SynapseNode::<f64>::pow_rank0(Rc::clone(&x),Rc::clone(&two));
				let (exp_sq_sn, b) = SynapseNode::<f64>::exp(a);
				let (sq_exp_sq_sn, y) = SynapseNode::<f64>::pow_rank0(b,two);

				fn func(x:f64) -> f64 {
					x.powf(2.0).exp().powf(2.0)
				}
				sq_exp_sq_sn.borrow().make_diff_node();
				exp_sq_sn.borrow().make_diff_node();
				square_sn.borrow().make_diff_node();
				let borrowed_x = x.borrow();
				if let Some(ref g) = borrowed_x.ref_grad() {
					let delta =  1.0e-4;
					let diff = (func(x0+delta)-func(x0-delta))/(2.0*delta);
					let dx = g.borrow().ref_signal()[vec![0,0]];
					assert!((diff-dx).abs() < delta);
				}
				else {
					println!("a.grad=None");
					assert!(false);
				}
			}
		}
	}

}
