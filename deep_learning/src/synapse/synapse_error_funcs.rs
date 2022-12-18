/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	fn mse_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
				   -> Vec<Tensor<T>> {
		let diff = inputs[0] - inputs[1];
		let one = num::one::<T>();
		let two = one+one;
		let diff_len:T =
			num::FromPrimitive::from_usize(diff.num_of_elements()).unwrap_or_else(|| panic!("invalid shape"));
		let mse = diff.pow(two).sum(&[1,1]).scale(one/diff_len);
		return vec![mse];
	}

	fn mse_backward(inputs: &Vec<NNNeuron<T>>,
				   grads: &Vec<NNNeuron<T>>,
				   _opt: &Option<SynapseOption>)
				   -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		(sns,outputs)
	}

	pub fn mean_square_error(x0:NNNeuron<T>,x1:NNNeuron<T>)
							 -> (NNSynapseNode<T>, NNNeuron<T>) {
		assert_eq!(x0.borrow().shape(), x1.borrow().shape());
		let label = "mean_square_error";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::mse_forward, Self::mse_backward);
		let sn = SynapseNode::<T>::new(&label, vec![Rc::clone(&x0),Rc::clone(&x1)], vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}
}
