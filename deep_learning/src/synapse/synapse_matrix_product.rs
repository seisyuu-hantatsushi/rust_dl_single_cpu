/* -*- tab-width:4 -*- */

use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	fn matrix_product_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
							  -> Vec<Tensor<T>> {
		vec![Tensor::<T>::matrix_product(inputs[0], inputs[1])]
	}

	fn matrix_product_backward(inputs: &Vec<NNNeuron<T>>,
							   grads: &Vec<NNNeuron<T>>,
							   _opt: &Option<SynapseOption>)
							   -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>){
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		if inputs[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
			return (sns,outputs);
		}
		outputs.push(Rc::clone(&grads[0]));

		if !inputs[0].borrow().is_constant() {
			let l_output = if grads[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
				let rt = inputs[1].borrow().ref_signal().transpose();
				let t = Tensor::<T>::matrix_product(grads[0].borrow().ref_signal(), &rt);
				let label = "(".to_string() + grads[0].borrow().name() + ") <*> (" + inputs[1].borrow().name() + "^t)";
				nn_neuron_constant(&label, t)
			}
			else {
				let (l_sn, l_output) = Self::transpose(Rc::clone(&inputs[1]));
				sns.push(l_sn);
				outputs.push(Rc::clone(&l_output));
				let (l_sn, l_output) = Self::matrix_product(Rc::clone(&grads[0]),l_output);
				sns.push(l_sn);
				l_output
			};

			{
				let mut left_neuron = inputs[0].borrow_mut();
				if let Some(ref g) = left_neuron.ref_grad(){
					outputs.push(Rc::clone(&g));
					let (sn, output) = Self::add(Rc::clone(&g), l_output);
					left_neuron.set_grad(Rc::clone(&output));
					sns.push(sn);
					outputs.push(output);
				}
				else {
					left_neuron.set_grad(l_output);
				}
			}
		}

		if !inputs[1].borrow().is_constant() {
			let r_output = if grads[0].borrow().is_constant() && inputs[0].borrow().is_constant() {
				let lt = inputs[0].borrow().ref_signal().transpose();
				let t = Tensor::<T>::matrix_product(&lt,grads[0].borrow().ref_signal());
				let label = "(".to_string() + inputs[0].borrow().name() + "^t) <*> (" + grads[0].borrow().name() + ")";
				nn_neuron_constant(&label, t)
			}
			else {
				let (r_sn, r_output) = Self::transpose(Rc::clone(&inputs[0]));
				sns.push(r_sn);
				outputs.push(Rc::clone(&r_output));
				let (r_sn, r_output) = Self::matrix_product(r_output,Rc::clone(&grads[0]));
				sns.push(r_sn);
				r_output
			};

			{
				let mut right_neuron = inputs[1].borrow_mut();
				if let Some(ref g) = right_neuron.ref_grad(){
					outputs.push(Rc::clone(&g));
					let (sn, output) = Self::add(Rc::clone(&g), r_output);
					right_neuron.set_grad(Rc::clone(&output));
					sns.push(sn);
					outputs.push(output);
				}
				else {
					right_neuron.set_grad(r_output);
				}
			}
		}

		(sns,outputs)
	}

	pub fn matrix_product(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>){
		let label = "matrix_product";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::matrix_product_forward,
								  Self::matrix_product_backward);
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x), Rc::clone(&y)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}
}
