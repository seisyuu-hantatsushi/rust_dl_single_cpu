/* -*- tab-width:4 -*- */
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	fn reshape_forward(inputs: Vec<&Tensor<T>>, opt: &Option<SynapseOption<T>>)
					   -> Vec<Tensor<T>> {
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
	}

	fn reshape_backward(inputs: &Vec<NNNeuron<T>>,
						grads:  &Vec<NNNeuron<T>>,
						opt: &Option<SynapseOption<T>>)
						-> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
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
	}

	pub fn reshape(x:NNNeuron<T>, dst_shape:Vec<usize>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let num_of_elements = x.borrow().ref_signal().buffer().len();
		let num_of_reshape_elements = dst_shape.iter().fold(1,|prod,d| prod * (*d));
		assert_eq!(num_of_elements, num_of_reshape_elements);
		let label = "reshape";
		let src_shape = x.borrow().ref_signal().shape().to_vec();
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&dst_shape));
		let s = Synapse::<T>::new_with_option(Self::reshape_forward,
											  Self::reshape_backward,
											  SynapseOption::Reshape((src_shape, dst_shape)));
		let sn = SynapseNode::<T>::new(&label,vec![Rc::clone(&x)],vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn transpose_forward(inputs: Vec<&Tensor<T>>, _: &Option<SynapseOption<T>>)
						 -> Vec<Tensor<T>> {
		vec![inputs[0].transpose()]
	}

	fn transpose_backward(inputs: &Vec<NNNeuron<T>>,
						  grads:  &Vec<NNNeuron<T>>,
						  _: &Option<SynapseOption<T>>)
						  -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
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
	}

	pub fn transpose(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "transpose";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   Synapse::<T>::new(Self::transpose_forward,
														 Self::transpose_backward));
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn sum_to_forward(inputs: Vec<&Tensor<T>>, opt: &Option<SynapseOption<T>>)
					  -> Vec<Tensor<T>> {
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
	}

	fn sum_to_backward(inputs: &Vec<NNNeuron<T>>,
					   grads:  &Vec<NNNeuron<T>>,
					   opt: &Option<SynapseOption<T>>)
					   -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
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
	}

	pub fn sum_to(x:NNNeuron<T>, shape:&[usize]) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "sum_to";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new_with_option(Self::sum_to_forward,
											  Self::sum_to_backward,
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

	fn broadcast_to_forward(inputs: Vec<&Tensor<T>>, opt: &Option<SynapseOption<T>>)
							-> Vec<Tensor<T>> {
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
	}

	fn broadcast_to_backward(inputs: &Vec<NNNeuron<T>>,
							 grads:  &Vec<NNNeuron<T>>,
							 opt: &Option<SynapseOption<T>>)
							 -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
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
	}

	pub fn broadcast_to(x:NNNeuron<T>, shape:&[usize]) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "broadcast_to";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let opt = SynapseOption::BroadcastTo((x.borrow().ref_signal().shape().to_vec(), shape.to_vec()));
		let s = Synapse::<T>::new_with_option(Self::broadcast_to_forward,
											  Self::broadcast_to_backward,
											  opt);

		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn slice_forward(inputs: Vec<&Tensor<T>>, opt: &Option<SynapseOption<T>>)
					 -> Vec<Tensor<T>> {
		let &index = if let Some(o) = opt {
			match o {
				SynapseOption::Slice((index, _)) => index,
				_ => panic!("invalid option")
			}
		}
		else {
			panic!("no option")
		};
		let sub_tensor = inputs[0].subtensor(index).into_tensor();
		vec![sub_tensor]
	}

	fn slice_backward(inputs: &Vec<NNNeuron<T>>,
					  grads:  &Vec<NNNeuron<T>>,
					  opt: &Option<SynapseOption<T>>)
					  -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		let (index, src_shape) = if let Some(ref o) = opt {
			match o {
				SynapseOption::Slice(s) => s,
				_ => panic!("invalid option")
			}
		}
		else {
			panic!("no option")
		};

		if !inputs[0].borrow().is_constant() {
			let (sn, output) = Self::slice_grad(Rc::clone(&inputs[0]),
												Rc::clone(&grads[0]),
												*index);
			outputs.push(Rc::clone(&inputs[0]));
			outputs.push(Rc::clone(&grads[0]));
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
	}

	pub fn slice(x:NNNeuron<T>, index:usize) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "slice";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let opt = SynapseOption::Slice((index, x.borrow().shape().to_vec()));
		let s = Synapse::<T>::new_with_option(Self::slice_forward,
											  Self::slice_backward,
											  opt);
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn slice_grad_forward(inputs: Vec<&Tensor<T>>, opt: &Option<SynapseOption<T>>)
						  -> Vec<Tensor<T>> {
		let &index = if let Some(ref o) = opt {
			match o {
				SynapseOption::SliceGrad(index) => {
					index
				},
				_ => panic!("invalid option")
			}
		}
		else {
			panic!("no option")
		};
		let dst_tensor = Tensor::<T>::zero(inputs[0].shape());
		vec![dst_tensor.add_at(&[index],inputs[1])]
	}

	fn slice_grad_backward(inputs: &Vec<NNNeuron<T>>,
						   grads:  &Vec<NNNeuron<T>>,
						   opt: &Option<SynapseOption<T>>)
						   -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		let &index = if let Some(ref o) = opt {
			match o {
				SynapseOption::SliceGrad(index) => {
					index
				},
				_ => panic!("invalid option")
			}
		}
		else {
			panic!("no option")
		};

		if !inputs[0].borrow().is_constant() {
			let (sn, output) = Self::slice(Rc::clone(&grads[0]),
										   index);
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
	}

	pub fn slice_grad(x:NNNeuron<T>, gx:NNNeuron<T>, index:usize) ->
		(NNSynapseNode<T>,NNNeuron<T>) {
		let label = "slice_grad";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let opt = SynapseOption::SliceGrad(index);
		let s = Synapse::<T>::new_with_option(Self::slice_grad_forward,
											  Self::slice_grad_backward,
											  opt);
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x),Rc::clone(&gx)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}
}
