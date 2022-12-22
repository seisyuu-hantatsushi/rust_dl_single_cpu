/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	fn hadamard_product_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
								-> Vec<Tensor<T>> {
		let left_shape  = inputs[0].shape().to_vec();
		let right_shape = inputs[1].shape().to_vec();
		let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
		let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });
		if left_prod < right_prod {
			let expand_left = inputs[0].broadcast(&right_shape);
			vec![Tensor::<T>::hadamard_product(&expand_left, inputs[1])]
		}
		else if left_prod > right_prod {
			let expand_right = inputs[1].broadcast(&left_shape);
			vec![Tensor::<T>::hadamard_product(inputs[0], &expand_right)]
		}
		else {
			vec![Tensor::<T>::hadamard_product(inputs[0], inputs[1])]
		}
	}

	fn hadamard_product_backward(inputs: &Vec<NNNeuron<T>>,
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
				let left_shape  = grads[0].borrow().ref_signal().shape().to_vec();
				let right_shape = inputs[1].borrow().ref_signal().shape().to_vec();
				let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
				let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });
				let t = if left_prod < right_prod {
					let expand_left = grads[0].borrow().ref_signal().broadcast(&right_shape);
					Tensor::<T>::hadamard_product(&expand_left, inputs[1].borrow().ref_signal())
				}
				else if left_prod > right_prod {
					let expand_right = inputs[1].borrow().ref_signal().broadcast(&left_shape);
					Tensor::<T>::hadamard_product(grads[0].borrow().ref_signal(), &expand_right)
				}
				else {
					Tensor::<T>::hadamard_product(grads[0].borrow().ref_signal(),
												  inputs[1].borrow().ref_signal())
				};

				let shaped_t = t.sum(inputs[0].borrow().ref_signal().shape());
				let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[1].borrow().name() + ")";
				nn_neuron_constant(&label, shaped_t)
			}
			else {
				let (l_sn, l_output) = Self::hadamard_product(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
				sns.push(l_sn);
				outputs.push(Rc::clone(&inputs[1]));
				outputs.push(Rc::clone(&l_output));
				let l_output = if l_output.borrow().ref_signal().shape() == inputs[0].borrow().ref_signal().shape() {
					let (l_sn, l_output) = Self::sum_to(l_output, inputs[0].borrow().ref_signal().shape());
					sns.push(l_sn);
					l_output
				}
				else {
					l_output
				};
				l_output
			};
			outputs.push(Rc::clone(&l_output));

			{
				let mut left_neuron = inputs[0].borrow_mut();
				if let Some(ref g) = left_neuron.ref_grad() {
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
				let left_shape  = inputs[0].borrow().ref_signal().shape().to_vec();
				let right_shape = grads[0].borrow().ref_signal().shape().to_vec();
				let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
				let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });

				let t = if left_prod < right_prod {
					let expand_left = inputs[0].borrow().ref_signal().broadcast(&right_shape);
					Tensor::<T>::hadamard_product(&expand_left, grads[0].borrow().ref_signal())
				}
				else if left_prod > right_prod {
					let expand_right = grads[0].borrow().ref_signal().broadcast(&left_shape);
					Tensor::<T>::hadamard_product(inputs[0].borrow().ref_signal(), &expand_right)
				}
				else {
					Tensor::<T>::hadamard_product(inputs[0].borrow().ref_signal(),
												  grads[0].borrow().ref_signal())
				};

				let shaped_t = t.sum(inputs[1].borrow().ref_signal().shape());
				let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[1].borrow().name() + ")";
				nn_neuron_constant(&label, shaped_t)
			}
			else {
				let (r_sn, r_output) = Self::hadamard_product(Rc::clone(&inputs[0]),Rc::clone(&grads[0]));
				sns.push(r_sn);
				outputs.push(Rc::clone(&inputs[0]));
				outputs.push(Rc::clone(&r_output));
				let r_output = if r_output.borrow().ref_signal().shape() != inputs[1].borrow().ref_signal().shape() {
					let (r_sn, r_output) = Self::sum_to(r_output, inputs[1].borrow().ref_signal().shape());
					sns.push(r_sn);
					r_output
				}
				else {
					r_output
				};
				r_output
			};

			{
				let mut right_neuron = inputs[1].borrow_mut();
				if let Some(ref g) = right_neuron.ref_grad() {
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

	pub fn hadamard_product(x:NNNeuron<T>,y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>){
		let label = "hadamard_product";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s =  Synapse::<T>::new(Self::hadamard_product_forward, Self::hadamard_product_backward);
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x), Rc::clone(&y)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn hadamard_division_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
								-> Vec<Tensor<T>> {
		let left_shape  = inputs[0].shape().to_vec();
		let right_shape = inputs[1].shape().to_vec();
		let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
		let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });
		if left_prod < right_prod {
			let expand_left = inputs[0].broadcast(&right_shape);
			vec![Tensor::<T>::hadamard_division(&expand_left, inputs[1])]
		}
		else if left_prod > right_prod {
			let expand_right = inputs[1].broadcast(&left_shape);
			vec![Tensor::<T>::hadamard_division(inputs[0], &expand_right)]
		}
		else {
			vec![Tensor::<T>::hadamard_division(inputs[0], inputs[1])]
		}
	}

	fn hadamard_division_backward(inputs: &Vec<NNNeuron<T>>,
								  grads: &Vec<NNNeuron<T>>,
								  _opt: &Option<SynapseOption>)
								  -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		if inputs[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
			return (sns,outputs);
		}
		outputs.push(Rc::clone(&grads[0]));

		if !inputs[0].borrow().is_constant() {
			let l_output = if grads[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
				let left_shape  = grads[0].borrow().ref_signal().shape().to_vec();
				let right_shape = inputs[1].borrow().ref_signal().shape().to_vec();
				let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
				let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });
				let t = if left_prod < right_prod {
					let expand_left = grads[0].borrow().ref_signal().broadcast(&right_shape);
					Tensor::<T>::hadamard_division(&expand_left, inputs[1].borrow().ref_signal())
				}
				else if left_prod > right_prod {
					let expand_right = inputs[1].borrow().ref_signal().broadcast(&left_shape);
					Tensor::<T>::hadamard_division(grads[0].borrow().ref_signal(), &expand_right)
				}
				else {
					Tensor::<T>::hadamard_division(grads[0].borrow().ref_signal(),
												   inputs[1].borrow().ref_signal())
				};

				let shaped_t = t.sum(inputs[0].borrow().ref_signal().shape());
				let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[1].borrow().name() + ")";
				nn_neuron_constant(&label, shaped_t)
			}
			else {
				let (l_sn, l_output) = Self::hadamard_division(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
				sns.push(l_sn);
				outputs.push(Rc::clone(&inputs[1]));
				outputs.push(Rc::clone(&l_output));
				let l_output = if l_output.borrow().ref_signal().shape() != inputs[0].borrow().ref_signal().shape() {
					let (l_sn, l_output) = Self::sum_to(l_output, inputs[0].borrow().ref_signal().shape());
					sns.push(l_sn);
					l_output
				}
				else {
					l_output
				};
				l_output
			};
			outputs.push(Rc::clone(&l_output));

			{
				let mut left_neuron = inputs[0].borrow_mut();
				if let Some(ref g) = left_neuron.ref_grad() {
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
			let two = nn_neuron_constant("2.0", (Tensor::<T>::one(&[1,1]))+Tensor::<T>::one(&[1,1]));
			outputs.push(Rc::clone(&two));
			let (neg, neg_input0) = Self::neg(Rc::clone(&inputs[0]));
			outputs.push(Rc::clone(&inputs[0]));
			sns.push(neg);
			outputs.push(Rc::clone(&neg_input0));
			let (r_sn, r_output) = Self::pow(Rc::clone(&inputs[1]),two);
			sns.push(r_sn);
			outputs.push(Rc::clone(&r_output));
			let (r_sn, r_output) = Self::hadamard_division(neg_input0, r_output);
			sns.push(r_sn);
			outputs.push(Rc::clone(&r_output));
			let (r_sn, r_output) = Self::hadamard_product(Rc::clone(&grads[0]), r_output);
			sns.push(r_sn);
			let r_output = if r_output.borrow().shape() != inputs[1].borrow().shape() {
				outputs.push(Rc::clone(&r_output));
				let (r_sn, r_output) = Self::sum_to(r_output, inputs[1].borrow().shape());
				sns.push(r_sn);
				r_output
			}
			else {
				r_output
			};
			outputs.push(Rc::clone(&r_output));
			{
				let mut right_neuron = inputs[1].borrow_mut();
				if let Some(ref g) = right_neuron.ref_grad() {
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

	pub fn hadamard_division(x:NNNeuron<T>,y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>){
		let label = "hadamard_division";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s =  Synapse::<T>::new(Self::hadamard_division_forward,
								   Self::hadamard_division_backward);
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
