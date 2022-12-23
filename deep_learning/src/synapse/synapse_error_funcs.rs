/* -*- tab-width:4 -*- */

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

		if inputs[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
			return (sns,outputs);
		}
		outputs.push(Rc::clone(&grads[0]));
		let (sn, diff) = Self::sub(Rc::clone(&inputs[0]), Rc::clone(&inputs[1]));
		sns.push(sn);
		outputs.push(Rc::clone(&diff));
		let (sn, output) = Self::broadcast_to(Rc::clone(&grads[0]), diff.borrow().shape());
		sns.push(sn);
		outputs.push(Rc::clone(&output));
		let diff_len:T = num::FromPrimitive::from_usize(diff.borrow().ref_signal().num_of_elements()).unwrap_or_else(|| panic!("invalid shape"));
		let two:T = num::one::<T>()+num::one::<T>();
		let (sn, gy) = Self::hadamard_product(output, diff);
 		sns.push(sn);
		outputs.push(Rc::clone(&gy));
		let scale = nn_neuron_constant("mes_backward_scale", Tensor::<T>::from_array(&[1,1],&[two/diff_len;1]));
		outputs.push(Rc::clone(&scale));
		let (sn, gy) = Self::hadamard_product(gy,scale);
		sns.push(sn);
		outputs.push(Rc::clone(&gy));
		if !inputs[0].borrow().is_constant() {
			let mut n = inputs[0].borrow_mut();
			if let Some(ref g) = n.ref_grad(){
				let (sn, output) = Self::sub(Rc::clone(&g), Rc::clone(&gy));
				sns.push(sn);
				outputs.push(output);
				n.set_grad(Rc::clone(&gy));
			}
			else {
				n.set_grad(Rc::clone(&gy));
			}
		}

		if !inputs[1].borrow().is_constant() {
			let (sn, neg_grad) = Self::neg(gy);
			sns.push(sn);
			outputs.push(Rc::clone(&neg_grad));
			let mut n = inputs[1].borrow_mut();
			if let Some(ref g) = n.ref_grad(){
				let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&neg_grad));
				sns.push(sn);
				outputs.push(Rc::clone(&output));
				n.set_grad(output);
			}
			else {
				n.set_grad(neg_grad);
			}
		}
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
