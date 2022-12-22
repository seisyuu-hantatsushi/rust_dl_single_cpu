/* -*- tab-width:4 -*- */
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	pub fn reshape(x:NNNeuron<T>, dst_shape:Vec<usize>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let num_of_elements = x.borrow().ref_signal().buffer().len();
		let num_of_reshape_elements = dst_shape.iter().fold(1,|prod,d| prod * (*d));
		assert_eq!(num_of_elements, num_of_reshape_elements);
		let label = "reshape";
		let src_shape = x.borrow().ref_signal().shape().to_vec();
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&dst_shape));
		let s = Synapse::<T>::new_with_option(
			|inputs, opt| {
				let dst_shape = if let Some(o) = opt {
					if let SynapseOption::Reshape((_s,d)) = o {
						d
					}
					else {
						panic!("Invalid Option")
					}
				}
				else {
					panic!("Invalid Option")
				};
				vec![inputs[0].reshape(&dst_shape)]
			},
			|inputs, grads, opt|{
				let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
				let mut outputs:Vec<NNNeuron<T>> = Vec::new();
				let (src_shape, _dst_shape) = if let Some(o) = opt {
					match o {
						SynapseOption::Reshape(s) => s,
						_ => panic!("invalid option")
					}
				}
				else {
					panic!("reshape must have option of itself")
				};

				if !inputs[0].borrow().is_constant() {
					let (sn,output) = Self::reshape(Rc::clone(&grads[0]), src_shape.clone());
					sns.push(sn);
					outputs.push(Rc::clone(&output));
					let mut n = inputs[0].borrow_mut();
					if let Some(ref g) = n.ref_grad() {
						outputs.push(Rc::clone(&g));
						let (sn, output) = Self::add(Rc::clone(&g), output);
						n.set_grad(Rc::clone(&output));
						sns.push(Rc::clone(&sn));
						outputs.push(output);
					}
					else {
						n.set_grad(output)
					}
				}
				(sns, outputs)
			},
			SynapseOption::Reshape((src_shape, dst_shape))
		);
		let sn = SynapseNode::<T>::new(&label,vec![Rc::clone(&x)],vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	pub fn transpose(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "transpose";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   Synapse::<T>::new(
										   |inputs, _opt| {
											   vec![inputs[0].transpose()]
										   },
										   |inputs, grads, _opt| {
											   let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											   let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											   if !inputs[0].borrow().is_constant() {
												   let (sn,output) = Self::transpose(Rc::clone(&grads[0]));
												   sns.push(sn);
												   outputs.push(Rc::clone(&output));
												   let mut n = inputs[0].borrow_mut();
												   if let Some(ref g) = n.ref_grad() {
													   let (sn,output) = Self::add(Rc::clone(&g), output);
													   sns.push(sn);
													   outputs.push(Rc::clone(&output));
													   n.set_grad(output);
												   }
												   else {
													   n.set_grad(output);
												   }
											   }
											   (sns,outputs)
										   }));
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	pub fn sum_to(x:NNNeuron<T>, shape:&[usize]) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "sum_to";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new_with_option(
			|inputs, opt| {
				if let Some(o) = opt {
					match o {
						SynapseOption::Sum((_src_shape, dst_shape)) => {
							vec![inputs[0].sum(&dst_shape)]
						},
						_ => panic!("invalid option")
					}
				}
				else {
					panic!("no option")
				}
			},
			|inputs, grads, opt| {
				let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
				let mut outputs:Vec<NNNeuron<T>> = Vec::new();
				let src_shape = if let Some(o) = opt {
					match o {
						SynapseOption::Sum((src_shape, _dst_shape)) => {
							src_shape
						},
						_ => panic!("invalid option")
					}
				}
				else {
					panic!("no option")
				};

				if !inputs[0].borrow().is_constant() {
					let (sn, output) = Self::broadcast_to(Rc::clone(&grads[0]), src_shape);
					sns.push(sn);
					outputs.push(Rc::clone(&output));
					let mut n = inputs[0].borrow_mut();
					if let Some(ref g) = n.ref_grad() {
						let (sn,output) = Self::add(Rc::clone(&g), output);
						sns.push(sn);
						outputs.push(Rc::clone(&output));
						n.set_grad(output);
					}
					else {
						n.set_grad(output);
					}
				}
				(sns,outputs)
			},
			SynapseOption::Sum((x.borrow().ref_signal().shape().to_vec(), shape.to_vec())));

		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   s);

		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	pub fn broadcast_to(x:NNNeuron<T>, shape:&[usize]) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "broadcast_to";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new_with_option(
			|inputs,opt| {
				let dst_shape = if let Some(o) = opt {
					match o {
						SynapseOption::BroadcastTo((_src_shape, dst_shape)) => {
							dst_shape
						},
						_ => panic!("invalid option")
					}
				}
				else {
					panic!("no option")
				};
				vec![inputs[0].broadcast(dst_shape)]
			},
			|inputs, grads, opt| {
				let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
				let mut outputs:Vec<NNNeuron<T>> = Vec::new();
				let src_shape = if let Some(o) = opt {
					match o {
						SynapseOption::BroadcastTo((src_shape, _dst_shape)) => {
							src_shape
						},
						_ => panic!("invalid option")
					}
				}
				else {
					panic!("no option")
				};

				if !inputs[0].borrow().is_constant() {
					let (sn, output) = Self::broadcast_to(Rc::clone(&grads[0]), src_shape);
					sns.push(sn);
					outputs.push(Rc::clone(&output));
					let mut n = inputs[0].borrow_mut();
					if let Some(ref g) = n.ref_grad() {
						let (sn,output) = Self::add(Rc::clone(&g), output);
						sns.push(sn);
						outputs.push(Rc::clone(&output));
						n.set_grad(output);
					}
					else {
						n.set_grad(output);
					}
				}
				(sns,outputs)
			},
			SynapseOption::BroadcastTo((x.borrow().ref_signal().shape().to_vec(), shape.to_vec())));

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
