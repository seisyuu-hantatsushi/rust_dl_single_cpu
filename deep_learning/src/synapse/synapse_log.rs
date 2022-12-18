/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron, nn_neuron_new, nn_neuron_constant};

impl<T> SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {

	fn ln_rank0_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
						-> Vec<Tensor<T>> {
		vec![inputs[0].ln()]
	}

	fn ln_rank0_backward(inputs: &Vec<NNNeuron<T>>,
						  grads: &Vec<NNNeuron<T>>,
						  _opt: &Option<SynapseOption>)
						 -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>){
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		// (ln x)' = 1/x
		let one = nn_neuron_constant("1.0", Tensor::<T>::one(&[1,1]));
		outputs.push(Rc::clone(&one));
		let (inverse_sn, inverse) = Self::div_rank0(one, Rc::clone(&inputs[0]));
		sns.push(inverse_sn);
		outputs.push(Rc::clone(&inverse));
		let (sn, output) = Self::mul_rank0(Rc::clone(&grads[0]), inverse);
		sns.push(sn);
		outputs.push(Rc::clone(&output));
		{
			let mut n = inputs[0].borrow_mut();
			if let Some(ref g) = n.ref_grad() {
				outputs.push(Rc::clone(&g));
				let (sn, output) = Self::add(Rc::clone(&g), output);
				n.set_grad(Rc::clone(&output));
				sns.push(sn);
				outputs.push(output);
			}
			else {
				n.set_grad(output);
			}
		}
		(sns,outputs)
	}

	pub fn ln_rank0(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		let label = "ln_rank0";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::ln_rank0_forward,
								  Self::ln_rank0_backward);
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn ln_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
						-> Vec<Tensor<T>> {
		vec![inputs[0].ln()]
	}

	fn ln_backward(inputs: &Vec<NNNeuron<T>>,
				   grads: &Vec<NNNeuron<T>>,
				   _opt: &Option<SynapseOption>)
				   -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		// (ln x)' = 1/x
		let one = nn_neuron_constant("1.0", Tensor::<T>::one(&[1,1]));
		outputs.push(Rc::clone(&one));
		let (inverse_sn, inverse) = Self::hadamard_division(one, Rc::clone(&inputs[0]));
		sns.push(inverse_sn);
		outputs.push(Rc::clone(&inverse));
		let (sn, output) = Self::hadamard_product(Rc::clone(&grads[0]), inverse);
		sns.push(sn);
		outputs.push(Rc::clone(&output));
		{
			let mut n = inputs[0].borrow_mut();
			if let Some(ref g) = n.ref_grad() {
				outputs.push(Rc::clone(&g));
				let (sn, output) = Self::add(Rc::clone(&g), output);
				n.set_grad(Rc::clone(&output));
				sns.push(sn);
				outputs.push(output);
			}
			else {
				n.set_grad(output);
			}
		}
		(sns,outputs)
	}

	pub fn ln(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		let label = "ln_rank0";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::ln_forward,
								  Self::ln_backward);
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

}
